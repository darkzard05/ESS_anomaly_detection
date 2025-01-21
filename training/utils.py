from src.Resnet_channel_spatial_attention import resnet18, resnet34, resnet50, resnet101, resnet152

def create_models(model_name='resnet18', num_classes=3):
    model_list = {'resnet18': resnet18,
                  'resnet34': resnet34,
                  'resnet50': resnet50,
                  'resnet101': resnet101,
                  'resnet152': resnet152
                  }
    # ResNet 모델 생성
    model = model_list[model_name](num_classes=num_classes, pretrained=False)
    
    return model # 생성된 모델 반환



