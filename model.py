import torch.nn as nn
import torchvision.models as models

def build_model(num_classes=2): 
    # 1. Load Standard DenseNet121
    model = models.densenet121(weights=None)
    
    # 2. Get input size (1024 for DenseNet121)
    num_ftrs = model.classifier.in_features
    
    # 3. Create the classifier with 2 outputs
    model.classifier = nn.Linear(num_ftrs, num_classes)
    
    return model
