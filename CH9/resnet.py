import torch
import torchvision

model = torchvision.models.resnet18()

example = torch.rand(1, 3, 224, 224)

traced_script_module = torch.jit.trace(model, example)

print(traced_script_module.graph)

traced_script_module.save("model.pt")