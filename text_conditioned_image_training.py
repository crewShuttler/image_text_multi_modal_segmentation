import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torch.nn as nn


CRACK_PROMPTS = [
    "segment crack",
    "segment wall crack",
    "segment the crack wall"
]

TAPING_PROMPTS = [
    "segment taping area",
    "segment drywall seam",
    "segment drywall joint",
    "segment the drywall taping area"
]

class DrywallDataset(Dataset):
    def __init__(self, image_dir, mask_dir, task_type):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)
        self.task_type = task_type

        self.transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.transform(image)
        mask = self.transform(mask)
        mask = (mask > 0).float()  # binary

        if self.task_type == "crack":
            prompt = random.choice(CRACK_PROMPTS)
        else:
            prompt = random.choice(TAPING_PROMPTS)

        return image, prompt, mask


import segmentation_models_pytorch as smp
from transformers import CLIPTextModel, CLIPTokenizer

class TextConditionedUNet(nn.Module):
    def __init__(self):
        super().__init__()

        # UNet backbone
        self.unet = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1
        )

        # CLIP text encoder
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        for param in self.unet.encoder.parameters():
            param.requires_grad = False

        self.text_proj = nn.Linear(512, 512)

    def forward(self, images, prompts):
        # Encode text
        tokens = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(images.device)
        with torch.no_grad():
            text_features = self.text_encoder(**tokens).pooler_output
        text_features = self.text_proj(text_features)
        
        with torch.no_grad():
        # Encode image
            img_features = self.unet.encoder(images)

        # Inject text into deepest feature map
        b, c, h, w = img_features[-1].shape
        text_features = text_features.unsqueeze(-1).unsqueeze(-1)
        text_features = text_features.expand(-1, -1, h, w)

        img_features[-1] = img_features[-1] + text_features

        # Decode
        masks = self.unet.decoder(img_features)
        masks = self.unet.segmentation_head(masks)

        return masks


def dice_loss(pred, target, smooth=1):
    pred = torch.sigmoid(pred)
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (pred * target).sum(1)
    dice = (2. * intersection + smooth) / (
        pred.sum(1) + target.sum(1) + smooth
    )
    return 1 - dice.mean()


def dice_score(pred, target, smooth=1e-6):
    pred = (torch.sigmoid(pred) > 0.5).float()
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (pred * target).sum(1)
    dice = (2 * intersection + smooth) / (pred.sum(1) + target.sum(1) + smooth)
    return dice.mean()


def iou_score(pred, target, smooth=1e-6):
    pred = (torch.sigmoid(pred) > 0.5).float()
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (pred * target).sum(1)
    union = pred.sum(1) + target.sum(1) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()



from torch.utils.data import DataLoader, ConcatDataset
import torch.optim as optim
import torchvision.utils as vutils


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("CUDA available:", torch.cuda.is_available())
    print("Device being used:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    crack_train = DrywallDataset(
        "dataset/crack/train/images",
        "dataset/crack/train/masks",
        task_type="crack"
    )

    taping_train = DrywallDataset(
        "dataset/taping/train/images",
        "dataset/taping/train/masks",
        task_type="taping"
    )

    train_dataset = ConcatDataset([crack_train, taping_train])

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Validation datasets
    crack_val = DrywallDataset(
        "dataset/crack/val/images",
        "dataset/crack/val/masks",
        task_type="crack"
    )

    taping_val = DrywallDataset(
        "dataset/taping/val/images",
        "dataset/taping/val/masks",
        task_type="taping"
    )

    val_dataset = ConcatDataset([crack_val, taping_val])

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    val_loader1 = DataLoader(
        crack_val,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    val_loader2 = DataLoader(
        taping_val,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )


    model = TextConditionedUNet().to(device)

    from torchinfo import summary

    s = summary(model.unet, input_size=(4, 3, 256, 256))
    print(s)
    # Save initial weights
    torch.save(model.state_dict(), "initial_weights.pth")
    print("Initial weights saved.")
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    bce = nn.BCEWithLogitsLoss()

    scaler = torch.amp.GradScaler("cuda")

    EPOCHS = 60

    output_dir = "debug_outputs"
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(EPOCHS):

        # -----------------------
        # TRAINING
        # -----------------------
        model.train()
        total_train_loss = 0

        print("Epoch Num Started:", epoch)

        for cnt, (images, prompts, masks) in enumerate(train_loader):

            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                preds = model(images, prompts)
                loss = bce(preds, masks) + dice_loss(preds, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()

            # Save only every 50 batches (avoid disk flood)
            if cnt % 50 == 0:
                with torch.no_grad():
                    preds_sigmoid = torch.sigmoid(preds)
                    preds_binary = (preds_sigmoid > 0.5).float()

                    input_img = images[0].detach().cpu()
                    gt_mask = masks[0].detach().cpu()
                    pred_mask = preds_binary[0].detach().cpu()

                    comparison = torch.cat([
                        input_img,
                        gt_mask.repeat(3,1,1),
                        pred_mask.repeat(3,1,1)
                    ], dim=2)

                    vutils.save_image(
                        comparison,
                        f"{output_dir}/epoch{epoch}_batch{cnt}.png"
                    )

        avg_train_loss = total_train_loss / len(train_loader)

        # -----------------------
        # VALIDATION
        # -----------------------
        model.eval()
        total_val_loss = 0
        total_dice = 0
        total_iou = 0

        with torch.no_grad():
            for images, prompts, masks in val_loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                with torch.amp.autocast("cuda"):
                    preds = model(images, prompts)
                    val_loss = bce(preds, masks) + dice_loss(preds, masks)

                total_val_loss += val_loss.item()
                total_dice += dice_score(preds, masks).item()
                total_iou += iou_score(preds, masks).item()

        avg_val_loss = total_val_loss / len(val_loader)
        avg_dice = total_dice / len(val_loader)
        avg_iou = total_iou / len(val_loader)

        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss:   {avg_val_loss:.4f}")
        print(f"Val Dice:   {avg_dice:.4f}")
        print(f"Val mIoU:   {avg_iou:.4f}")

        total_val_loss1 = 0
        total_dice1 = 0
        total_iou1 = 0

        with torch.no_grad():
            for images, prompts, masks in val_loader1:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                with torch.amp.autocast("cuda"):
                    preds = model(images, prompts)
                    val_loss = bce(preds, masks) + dice_loss(preds, masks)

                total_val_loss1 += val_loss.item()
                total_dice1 += dice_score(preds, masks).item()
                total_iou1 += iou_score(preds, masks).item()

        avg_val_loss = total_val_loss1 / len(val_loader1)
        avg_dice = total_dice1 / len(val_loader1)
        avg_iou = total_iou1 / len(val_loader1)

        #print(f"\nEpoch {epoch+1}")
        #print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss_crack:   {avg_val_loss:.4f}")
        print(f"Val Dice_crack:   {avg_dice:.4f}")
        print(f"Val mIoU_crack:   {avg_iou:.4f}")

        total_val_loss2 = 0
        total_dice2 = 0
        total_iou2 = 0

        with torch.no_grad():
            for images, prompts, masks in val_loader2:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                with torch.amp.autocast("cuda"):
                    preds = model(images, prompts)
                    val_loss = bce(preds, masks) + dice_loss(preds, masks)

                total_val_loss2 += val_loss.item()
                total_dice2 += dice_score(preds, masks).item()
                total_iou2 += iou_score(preds, masks).item()

        avg_val_loss = total_val_loss2 / len(val_loader2)
        avg_dice = total_dice2 / len(val_loader2)
        avg_iou = total_iou2 / len(val_loader2)

        #print(f"\nEpoch {epoch+1}")
        #print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss_taping:   {avg_val_loss:.4f}")
        print(f"Val Dice_taping:   {avg_dice:.4f}")
        print(f"Val mIoU_taping:   {avg_iou:.4f}")
        print("-" * 50)
        if epoch % 4 == 0 :
            torch.save(model.state_dict(), f"{output_dir}/epoch{epoch}.pth")

if __name__ == "__main__":
    main()
