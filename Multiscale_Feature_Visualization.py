import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import save_image

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def pil_loader_gray(path):
    img = Image.open(path)
    img = img.convert("L")
    return img

def cosine_similarity_map(loaded_tensor, name, idx, idx2):
    if idx2 =='02':
        if idx == 1:
            reference_vector = loaded_tensor[0, :, 16, 253]  # 形状 (64,)
        if idx == 2:
            reference_vector = loaded_tensor[0, :, 7, 126]  # 形状 (64,)
        if idx == 3:
            reference_vector = loaded_tensor[0, :, 3, 63]  # 形状 (64,)
        if idx == 4:
            reference_vector = loaded_tensor[0, :, 1, 31]  # 形状 (64,)
    if idx2 == '08':
        if idx == 1:
            reference_vector = loaded_tensor[0, :, 56, 220]  # 形状 (64,)
        if idx == 2:
            reference_vector = loaded_tensor[0, :, 28, 110]  # 形状 (64,)
        if idx == 3:
            reference_vector = loaded_tensor[0, :, 14, 55]  # 形状 (64,)
        if idx == 4:
            reference_vector = loaded_tensor[0, :, 1, 31]  # 形状 (64,)
    # 对张量中的每个向量计算余弦相似度
    # 需要将参考向量扩展维度以便进行广播操作
    reference_vector = reference_vector.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # 形状 (1, 64, 1, 1)

    # 计算余弦相似度
    cosine_similarity = F.cosine_similarity(loaded_tensor, reference_vector, dim=1, eps=1e-8)
    # cosine_similarity的尺寸为torch.Size([1, 96, 320])
    cosine_similarity = cosine_similarity.unsqueeze(0)  # 形状 (1, 3, 96, 320)
    # print(cosine_similarity.shape)  # 输出: torch.Size([1, 1, 96, 320])
    save_image(cosine_similarity, "Multiscale_Feature/{}_{}_test.png".format(idx2, name))

loaded_tensor = torch.load('Multiscale_Feature/02_d1_1.pt')
cosine_similarity_map(loaded_tensor[:, :18], "d1_1", 2,'02')
