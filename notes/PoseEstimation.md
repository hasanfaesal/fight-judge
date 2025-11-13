I've finetuned YOLO11x-pose on my custom dataset linked in the README.md
The initial results are promising, but there is still room for improvement.

Potentially, I can shift to a model like RTMO which uses one-stage pose estimation and does faster inference allowing me to achieve 'real-time' inference. Real time however isn't well defined and can mean different things in different contexts.

Options like ViTPose and DETRPose also look promising but with the compromise of inference speed.