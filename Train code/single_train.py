import torch
from test import operate_pic
from Utils import Option
from tensorboardX import SummaryWriter
import torch.optim as optim
from SubEncoder_Unet import DE_Unet
from Unet import Unet

device = torch.device('cuda:0')
test_image_path = r"C:\Users\guany\PycharmProjects\UDC\UDC_train\Train\Toled\LQ\2.png"
label_image_path = r"C:\Users\guany\PycharmProjects\UDC\UDC_train\Train\Toled\HQ\2.png"

writer = SummaryWriter()
# model=Unet().cuda()
# model=torch.load(Option.UNet_model_save_path)
# model=DE_Unet().cuda()
model=torch.load(Option.DE_UNet_model_save_path)
input_tensor = operate_pic(test_image_path)
input_tensor = input_tensor.to(device)
label_tensor = operate_pic(label_image_path)
label_tensor = label_tensor.to(device)

# print(input_tensor.size(),label_tensor.size())
# writer.add_images("label" , label_tensor)
# writer.add_images("input" , input_tensor)

criterion = torch.nn.L1Loss().cuda()
optimizer = optim.Adam(
    model.parameters(),
    lr=Option.learning_rate,
    weight_decay=0.5)


for epoch in range(1, 300):
    optimizer.zero_grad()
    output = model(input_tensor)
    # writer.add_images("output" + str(epoch), output)
    loss = criterion(output, label_tensor)
    print(loss)
    loss.backward()
    optimizer.step()
    writer.add_scalar("loss", loss, epoch)
torch.save(model, Option.DE_UNet_model_save_path)