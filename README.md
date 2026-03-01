# image_text_multi_modal_segmentation


Network architectures used - in progress

Unet encoder for image data processing and embedding prediction
ResNet-34 architecture for Unet encoder initialized with pretrained weights created using Imagenet dataset

CLIP text tokenizer for converting raw text sequence into an text embedding vector.
CLIP text model for text sequence processing and text features estimation

Unet decoder for fused image and text features processing to estimate binary segmentation estimation

Datasets used explanation:
5100 - crack image segmentation training
202 - crack image segementation validation

850 - taping image segmentation training
201 - taping image segementation validation

Text Data
randomly assigned from a sequence of texts created related to taping segment request and crack segment request

Images output data
Polygon, bounding box segmentation pixel annotation are processed using pre-processing scripts (.py) to generate binary images as ground truth outputs

Training methodology of model:

iteration 1:
Trained for 50 epochs with batch size 8, learing rate 1e-4
By freezing parameters of CLIPText Model


iteration 2:
Trained for 50 epochs with batch size 12, learning rate 1e-5
By freezing parameters of CLIPText Model


iteraion 3:
Trained for 50 epochs with batch size 8, learing rate 1e-5
By freezing parameters of Unet encoder
By freezing parameters of CLIPText Model


Validation methodology of model:

Average training / inference time per image:
45 ms, 23 ms

Failure scenarios:
Complex segmentation structures 

Error functions:

DICE, BCE

Metrics for validation:
DICE score, mIOU score

Precision/recall

TP, FP, FN, TN in validation data

Confusion matrix:

Model weights:

