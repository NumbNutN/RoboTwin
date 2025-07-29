import cv2
import torch
import numpy as np
import torchvision
from PIL import Image, ImageOps


# 将增强函数移到类外部
def enhance_black_regions(img):
    """Enhance black regions in the image by adjusting contrast in dark areas."""
    # Extract V channel
    v = img.split()[-1] 
    
    # Adaptive threshold processing
    threshold = np.percentile(np.array(v), 10)
    mask = np.array(v) < threshold
    
    # Enhance contrast
    v = v.point(lambda p: p*1.5 if p < threshold else p)
    
    # Merge channels
    h,s,_ = img.split()
    return Image.merge("HSV", (h,s,v))


def randomize_bright_regions(img):
    """
    保留图像中最暗的45%像素（最黑的部分），其余较亮的像素用随机值替换
    """
    # 转换为NumPy数组和灰度图
    img_array = np.array(img)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # 找出前45%最暗的像素阈值
    threshold = np.percentile(gray, 45)
    
    # 创建掩码：灰度值小于等于阈值的部分为True（保留暗区域）
    mask = gray <= threshold
    
    # 生成随机值替换非暗区域
    h, w = gray.shape
    c = img_array.shape[2]  # 通道数
    random_values = np.random.randint(0, 256, (h, w, c), dtype=np.uint8)
    
    # 复制原始图像
    result = img_array.copy()
    
    # 应用掩码：保留暗区域，替换亮区域
    for i in range(c):
        channel = result[:,:,i]
        channel[~mask] = random_values[:,:,i][~mask]
        result[:,:,i] = channel
    
    return Image.fromarray(result)


class DinoPreprocessor:
    """Handles image preprocessing for DINO model including augmentation and normalization."""
    
    # 渐进增强配置（progress: 0~1）
    AUG_CONFIG = {
        # 参数格式: (初始值, 最终值)，初始值是原先采用的方案
        'brightness_range': ((0.8, 1.2), (0.4, 1.6)),
        'contrast_range': ((0.7, 1.3), (0.7, 1.7)),
        'saturation_range': ((0.5, 1.5), (0.5, 2.0)),
        'hue_shift': (0.05, 0.10),
        'random_apply_prob': (0.4, 0.8),  # TODO: the percentile of randomize_bright_regions can also be randomized
        'sharpness_factor': (1.8, 2.2),
        'sharpness_prob': (0.7, 0.9)
    }
    
    def __init__(self, args):
        self.use_transform = args.use_transform
        self.current_progress = 0.0  # 当前训练进度 0~1
        
        self.init_transforms()
    
    def init_transforms(self):
        # Determine image dimensions based on camera configuration
        self.height = 720  # 480 + 240
        self.width = 640
        self.dino_size = 518
        
        # 初始化transform
        self.transform = self._build_transform()
        
        # DINO normalization
        self.normalize_dino = torchvision.transforms.Compose([
            torchvision.transforms.Resize((self.dino_size, self.dino_size)),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            
            # 灰度化后的归一化参数
            # torchvision.transforms.Normalize(
            #     mean=[0.38981062173843384, 0.38981062173843384, 0.38981062173843384], 
            #     std=[0.1762208342552185, 0.1762208342552185, 0.1762208342552185]    
            # ),
        ])
    
    def _lerp(self, start, end, progress):
        """线性插值"""
        return start + (end - start) * progress

    def _build_transform(self):
        """根据当前进度构建增强transform"""
        p = self.current_progress
        
        # 计算当前参数值
        params = {}
        for key, (start, end) in self.AUG_CONFIG.items():
            if isinstance(start, tuple):  # 处理范围值
                params[key] = (
                    self._lerp(start[0], end[0], p),
                    self._lerp(start[1], end[1], p)
                )
            else:  # 处理单值
                params[key] = self._lerp(start, end, p)
        
        return torchvision.transforms.Compose([
            torchvision.transforms.Resize((self.height, self.width)),
            torchvision.transforms.ColorJitter(
                brightness=params['brightness_range'],
                contrast=params['contrast_range'],
                saturation=params['saturation_range'],
                hue=params['hue_shift']
            ),
            # # Black region enhancement
            # torchvision.transforms.Lambda(lambda x: enhance_black_regions(x)),
            # # 随机增强亮区域（背景）
            # torchvision.transforms.RandomApply([
            #     torchvision.transforms.Lambda(randomize_bright_regions)
            # ], p=params['random_apply_prob']),
            # # 锐化
            # torchvision.transforms.RandomAdjustSharpness(
            #     sharpness_factor=params['sharpness_factor'],
            #     p=params['sharpness_prob']
            # ),
            # # 灰度化
            # torchvision.transforms.Lambda(lambda x: x.convert("L")),      # Convert to grayscale
            # torchvision.transforms.Lambda(lambda x: ImageOps.autocontrast(x)), # Adaptive contrast
            # torchvision.transforms.Lambda(lambda x: x.convert("RGB")),    # Convert back to RGB
        ])

    def set_augmentation_progress(self, progress):
        """更新训练进度（0.0~1.0）并重建transform"""
        self.current_progress = max(0.0, min(1.0, progress))
        self.transform = self._build_transform()
    
    def process_image(self, image):
        """Process a single image for DINO model input. input shape is [H, W, C]"""
        # 如果输入是numpy数组（来自旧代码路径）
        if isinstance(image, np.ndarray):
            # 转换为PIL图像
            image = Image.fromarray(image)
        
        # 如果输入是PIL图像
        if isinstance(image, Image.Image):
            # 应用图像增强
            if self.use_transform:
                image = self.transform(image)
            # image: [H, W, C], 0-255
            # 转换为tensor并归一化
            image = torchvision.transforms.functional.to_tensor(image).float()
            # image shape: [C, H, W], 0-1
        
        # 应用DINO标准化
        return self.normalize_dino(image)
    
    def process_batch(self, images, pos=None):
        """Process a batch of images and optionally handle position data for DINO model."""
        # For standard mode, just process the RGB images
        processed_images = torch.stack([self.process_image(img) for img in images])
        
        return processed_images
    
    def handle_flip(self, images, pos):
        """Handle flipping of images and position data. FOR SINGLE IMAGE AND POS!"""
        if isinstance(images, Image.Image):
            # 水平翻转PIL图像
            flipped_images = images.transpose(Image.FLIP_LEFT_RIGHT)
        
        elif isinstance(images, torch.Tensor):
            # Flip images horizontally
            flipped_images = torch.flip(images, [2])
        
        # Create flipped position vector
        flipped_pos = torch.zeros_like(pos)
        flipped_pos[:7] = pos[7:]
        flipped_pos[7:] = pos[:7]
        
        # Negate specific components that need to be reversed
        flipped_pos[[0, 4, 5, 7, 11, 12]] *= -1
        
        return flipped_images, flipped_pos
    
    def flip_images_pos_batch(self, images, pos):
        """
        randomly flip images and positions, images shape is [B, 3, H, W], pos shape is [B, 14]
        """
        flip_flags = torch.randint(0, 2, (len(images), 1))
        # Handle flipping if needed
        if flip_flags.any():
            # Only flip the images and positions where flip_flag is True
            for i, flip in enumerate(flip_flags):
                if flip:
                    images[i], pos[i] = self.handle_flip(images[i], pos[i])
        return images, pos


# -------------------------- 分割机械臂 --------------------------

def segment_robot_arms(image, left_coord=(0.201, 0.378), right_coord=(0.791, 0.374), color_diff=30):
    """
    Segment and split left and right robotic arms in the image using region growing.
    
    Args:
        image: Input image (BGR format, but class Dataset uses RGB format), numpy array. But RGB seems to be better.
        left_coord: Normalized coordinates for the left arm seed point (x, y)
        right_coord: Normalized coordinates for the right arm seed point (x, y)
        color_diff: Color difference threshold for region growing
        
    Returns:
        result_image: Image with marked arms and bounding boxes
        arm_boxes: Dictionary with 'left' and 'right' bounding boxes (x, y, w, h)
    """
    # image is a PIL Image
    if isinstance(image, Image.Image):
        image = np.array(image)
    h, w = image.shape[:2]
    
    # Convert normalized coordinates to pixel coordinates
    left_x = int(left_coord[0] * w)
    left_y = int(left_coord[1] * h)
    right_x = int(right_coord[0] * w)
    right_y = int(right_coord[1] * h)
    
    def grow_region(img, seed, diff):
        """Region growing algorithm with edge case handling"""
        mask = np.zeros((h+2, w+2), dtype=np.uint8)
        seed_color = img[seed[1], seed[0]]
        flags = cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | 255 << 8
        try:
            cv2.floodFill(
                image=img,
                mask=mask,
                seedPoint=seed,
                newVal=255,
                loDiff=(diff, diff, diff),
                upDiff=(diff, diff, diff),
                flags=flags
            )
        except:
            return np.zeros((h, w), dtype=np.uint8)
        return mask[1:-1, 1:-1]
    
    # Generate masks with error handling
    mask_left = grow_region(image.copy(), (left_x, left_y), color_diff)
    mask_right = grow_region(image.copy(), (right_x, right_y), color_diff)
    
    # Validity checking function
    def check_validity(mask, seed, is_left):
        if mask.sum() == 0:
            return False
        
        # Check seed containment
        if mask[seed[1], seed[0]] == 0:
            return False
        
        # Check spatial constraints
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return False
        
        main_contour = max(contours, key=cv2.contourArea)
        x, y, wc, hc = cv2.boundingRect(main_contour)
        
        # Check if crosses center line
        if is_left and (x + wc) > w * 0.6:  # Allow 60% of width
            return False
        if not is_left and x < w * 0.4:     # Allow 40% of width
            return False
        
        return True
    
    # Check validity and overlap
    left_valid = check_validity(mask_left, (left_x, left_y), True)
    right_valid = check_validity(mask_right, (right_x, right_y), False)
    overlap = cv2.bitwise_and(mask_left, mask_right)
    has_overlap = cv2.countNonZero(overlap) > 0
    
    # Create visualization and bounding boxes
    result = image.copy()
    arm_boxes = {'left': None, 'right': None, 'split_line': w//2}
    
    # Process left arm
    if left_valid and not has_overlap:
        contours, _ = cv2.findContours(mask_left, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            x, y, wc, hc = cv2.boundingRect(cnt)
            arm_boxes['left'] = (x, y, wc, hc)
            cv2.rectangle(result, (x, y), (x+wc, y+hc), (0, 0, 255), 2)
    
    # Process right arm
    if right_valid and not has_overlap:
        contours, _ = cv2.findContours(mask_right, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            x, y, wc, hc = cv2.boundingRect(cnt)
            arm_boxes['right'] = (x, y, wc, hc)
            cv2.rectangle(result, (x, y), (x+wc, y+hc), (0, 255, 0), 2)
    
    # Calculate split line
    left_split = int(3*w/5) if not left_valid or has_overlap else (arm_boxes['left'][0] + arm_boxes['left'][2])
    right_split = int(2*w/5) if not right_valid or has_overlap else arm_boxes['right'][0]
    if left_split < right_split:
        arm_boxes['left_split'] = arm_boxes['right_split'] = (left_split + right_split) // 2
    else:
        arm_boxes['left_split'] = left_split
        arm_boxes['right_split'] = right_split
    # Draw split line
    cv2.line(result, (arm_boxes['left_split'], 0), (arm_boxes['left_split'], 2*h//3), (255, 0, 0), 2)
    cv2.line(result, (arm_boxes['right_split'], 0), (arm_boxes['right_split'], 2*h//3), (255, 0, 0), 2)
    
    # 2/3*h split horizon line
    cv2.line(result, (0, 2*h//3), (w, 2*h//3), (255, 0, 0), 2)
    arm_boxes['arm_gripper_split'] = int(2*h//3)
    # left gripper and right gripper split
    arm_boxes['gripper_split'] = int(w//2)
    cv2.line(result, (arm_boxes['gripper_split'], 2*h//3), (arm_boxes['gripper_split'], h), (255, 0, 0), 2)
    return result, arm_boxes
