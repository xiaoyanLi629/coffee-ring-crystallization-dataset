import os
import cv2
import numpy as np
import shutil
from pathlib import Path
import argparse

class ImageCorrector:
    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
    def correct_normal_image(self, image_path):
        """
        校正普通手机拍摄图像的白平衡和亮度
        """
        img = cv2.imread(str(image_path))
        if img is None:
            return None
            
        # 转换到LAB色彩空间进行白平衡校正
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 应用CLAHE (Contrast Limited Adaptive Histogram Equalization) 到L通道
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # 白平衡校正 - Gray World算法
        # 计算a和b通道的平均值并校正
        a_mean = np.mean(a)
        b_mean = np.mean(b)
        
        # 将a和b通道向中性点(128)调整
        a_offset = int(128 - a_mean)
        b_offset = int(128 - b_mean)
        a = cv2.add(a, a_offset)
        b = cv2.add(b, b_offset)
        
        # 合并通道并转换回BGR
        lab_corrected = cv2.merge([l, a, b])
        corrected = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)
        
        # 应用gamma校正来调整整体亮度
        gamma = 1.2
        lookup_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
        corrected = cv2.LUT(corrected, lookup_table)
        
        return corrected
    
    def correct_sem_image(self, image_path):
        """
        校正SEM图像的对比度和亮度
        """
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
            
        # 应用CLAHE增强对比度
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(img)
        
        # 直方图均衡化
        equalized = cv2.equalizeHist(enhanced)
        
        # 组合原图和增强图像以保持细节
        result = cv2.addWeighted(enhanced, 0.7, equalized, 0.3, 0)
        
        # 降噪
        denoised = cv2.bilateralFilter(result, 9, 75, 75)
        
        return denoised
    
    def should_keep_file(self, filename):
        """
        判断文件是否应该保留
        """
        # 保留的文件类型
        keep_files = {
            'image.jpg',  # 普通图像
            'sem.jpg',    # SEM图像
            'sem.txt',    # SEM元数据
            'C.jpeg',     # 碳元素EDS
            'O.jpeg',     # 氧元素EDS
            'Na.jpeg',    # 钠元素EDS
            'Mg.jpeg',    # 镁元素EDS
            'S.jpeg',     # 硫元素EDS
            'Cl.jpeg',    # 氯元素EDS
            'Ca.jpeg'     # 钙元素EDS
        }
        
        return filename in keep_files
    
    def process_directory(self, rel_path=""):
        """
        递归处理目录结构
        """
        current_input = self.input_dir / rel_path
        current_output = self.output_dir / rel_path
        
        if not current_input.exists():
            return
            
        # 创建输出目录
        current_output.mkdir(parents=True, exist_ok=True)
        
        # 处理当前目录中的文件
        for item in current_input.iterdir():
            if item.is_file():
                if self.should_keep_file(item.name):
                    output_file = current_output / item.name
                    
                    if item.name == 'image.jpg':
                        # 校正普通图像
                        print(f"Correcting normal image: {rel_path}/{item.name}")
                        corrected = self.correct_normal_image(item)
                        if corrected is not None:
                            cv2.imwrite(str(output_file), corrected)
                        else:
                            print(f"Failed to process: {item}")
                            
                    elif item.name == 'sem.jpg':
                        # 校正SEM图像
                        print(f"Correcting SEM image: {rel_path}/{item.name}")
                        corrected = self.correct_sem_image(item)
                        if corrected is not None:
                            cv2.imwrite(str(output_file), corrected)
                        else:
                            print(f"Failed to process: {item}")
                            
                    elif item.name == 'sem.txt' or item.name.endswith('.jpeg'):
                        # 直接复制EDS图像和元数据文件
                        print(f"Copying file: {rel_path}/{item.name}")
                        shutil.copy2(item, output_file)
                        
            elif item.is_dir():
                # 递归处理子目录
                new_rel_path = rel_path / item.name if rel_path else Path(item.name)
                self.process_directory(new_rel_path)
    
    def process_all(self):
        """
        处理所有表格的数据
        """
        print(f"Starting image correction process...")
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        
        # 处理所有Table目录
        for table_dir in self.input_dir.glob("Table_*"):
            if table_dir.is_dir():
                rel_path = Path(table_dir.name)
                print(f"Processing {table_dir.name}...")
                self.process_directory(rel_path)
        
        print("Image correction process completed!")

def main():
    parser = argparse.ArgumentParser(description='Correct SEM and normal images brightness and white balance')
    parser.add_argument('--input', '-i', default='data', 
                       help='Input directory containing the original data (default: data)')
    parser.add_argument('--output', '-o', default='data_corrected', 
                       help='Output directory for corrected data (default: data_corrected)')
    
    args = parser.parse_args()
    
    # 创建图像校正器实例
    corrector = ImageCorrector(args.input, args.output)
    
    # 处理所有图像
    corrector.process_all()

if __name__ == "__main__":
    main() 