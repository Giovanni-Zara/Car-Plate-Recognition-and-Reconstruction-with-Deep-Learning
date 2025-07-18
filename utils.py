from imports import *  # Importing all necessary libraries
from global_vars import *  # Importing global variables for license plate processing


def ctc_collate_fn(batch):
    '''
    basically what I do here is stacking all the images     def create_annotation_files(self):
        for img_name in self.image_names:
            x_center, y_center, width, height = self.get_bbox_coords_YOLO_format(img_name)
            class_id = 0  # Assuming license plate is the only class

            annotation_file = os.path.join(self.output_dir, f"{os.path.splitext(img_name)[0]}.txt")
            
            with open(annotation_file, 'w') as f:
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
    
    def visualize_bbox(self, filename, save_path=None):
        """
        Visualize the bounding box on the resized image to verify correctness
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        # Load the resized image
        img_path = os.path.join(self.img_dir, filename)
        image = Image.open(img_path)
        
        # Get YOLO coordinates
        x_center, y_center, width, height = self.get_bbox_coords_YOLO_format(filename)
        
        # Convert YOLO format back to pixel coordinates for visualization
        img_width, img_height = image.size
        x_center_px = x_center * img_width
        y_center_px = y_center * img_height
        width_px = width * img_width
        height_px = height * img_height
        
        # Calculate top-left corner for rectangle
        x_min = x_center_px - width_px / 2
        y_min = y_center_px - height_px / 2
        
        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.imshow(image)
        
        # Add bounding box
        rect = patches.Rectangle((x_min, y_min), width_px, height_px,
                               linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        
        # Add center point
        ax.plot(x_center_px, y_center_px, 'ro', markersize=8)
        
        ax.set_title(f'Bounding Box Visualization\n{filename}')
        ax.set_xlabel(f'YOLO coords: ({x_center:.3f}, {y_center:.3f}, {width:.3f}, {height:.3f})')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        plt.close()h into a single tensor and
    then computing the len of each label (assuming different lenght plate can happen).
    Finally just concatenating all the labels into a vector (pytorch CTC wantres them in a line, not list)
    then returning image-label-its lenght.
    I need this to tell CTC where labels finish and i do not care padding as CTC deals with that internally (NICE)

    '''
    images, labels = zip(*batch)
    images = torch.stack(images)
    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    labels = torch.cat(labels)
    return images, labels, label_lengths


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for images, labels, label_lengths in dataloader:
        # Validation checks
        if images.size(0) == 0:
            print("Warning: Empty batch encountered, skipping...")
            continue
            
        images = images.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)  # [W, B, num_classes]
        log_probs = outputs.log_softmax(2)
        input_lengths = torch.full(size=(images.size(0),), fill_value=log_probs.size(0), dtype=torch.long).to(device)
        
        # Check for valid inputs to CTC
        if torch.any(input_lengths < label_lengths):
            print("Warning: Input length < label length, adjusting...")
            input_lengths = torch.clamp(input_lengths, min=label_lengths.max())
            
        loss = criterion(log_probs, labels, input_lengths, label_lengths)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
        
    avg_loss = total_loss / max(num_batches, 1)  # Avoid division by zero
    print(f"Epoch average loss: {avg_loss:.4f}")
    return avg_loss


def ctc_greedy_decoder(output, idx2char, blank=BLANK_IDX):
    '''
    Now, I know the network returns probabilities, as it does a softmax with logits of characters.
    I need to transform that probability into an actual char to compose the plate.
    I take the argmax of the softmax (most prob char), remove blanks used by CTC and possible
    duplicates CTC can actually produce.
    At the end I simply use the  mappings char-index index-char deefined at the beginning to compose the plate.
    This is greedy as it just takes the argmax of every step, I think it's more than enough here.
    '''
    # output: [seq_len, batch, num_classes]
    out = output.permute(1, 0, 2)  # [batch, seq_len, num_classes]
    pred_strings = []
    for probs in out:
        pred = probs.argmax(1).cpu().numpy()
        #
        prev = -1
        pred_str = []
        for p in pred:
            if p != blank and p != prev:
                pred_str.append(idx2char[p])
            prev = p
        pred_strings.append(''.join(pred_str))
    return pred_strings


'''def evaluate(model, dataloader, device):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():   #classic stuff to speed it up, no need to compute gradients
        for images, labels, label_lengths in dataloader:
            images = images.to(device)
            outputs = model(images)
            pred_strings = ctc_greedy_decoder(outputs, idx2char)
            # Ricostruisci le targhe vere
            labels_cpu = labels.cpu().numpy()
            lengths_cpu = label_lengths.cpu().numpy()
            idx = 0
            gt_strings = []
            for l in lengths_cpu:
                gt = ''.join([idx2char[i] for i in labels_cpu[idx:idx+l]])
                gt_strings.append(gt)
                idx += l
            for pred, gt in zip(pred_strings, gt_strings):
                if pred == gt:
                    correct += 1
                total += 1
    acc = correct / total
    print(f"Eval accuracy (full plate): {acc:.4f}")
    return acc'''

def evaluate(model, dataloader, device):
    model.eval()
    total = 0
    correct = 0
    total_chars = 0
    correct_chars = 0
    with torch.no_grad():
        for images, labels, label_lengths in dataloader:
            images = images.to(device)
            outputs = model(images)
            pred_strings = ctc_greedy_decoder(outputs, idx2char)
            labels_cpu = labels.cpu().numpy()   #need to move tensors tu cpu memory for numpy
            lengths_cpu = label_lengths.cpu().numpy()
            idx = 0
            gt_strings = []
            for l in lengths_cpu:
                gt = ''.join([idx2char[i] for i in labels_cpu[idx:idx+l]])
                gt_strings.append(gt)
                idx += l
            for pred, gt in zip(pred_strings, gt_strings):
                if pred == gt:
                    correct += 1
                total += 1
                # Character-level accuracy, had to introduce this cause train is very slow.
                # Need to know if at least model's getting some chars
                min_len = min(len(pred), len(gt))
                if min_len > 0:  # Avoid empty strings
                    correct_chars += sum([p == g for p, g in zip(pred[:min_len], gt[:min_len])])
                total_chars += len(gt)  # Always count ground truth characters
    acc = correct / total
    acc_char = correct_chars / total_chars
    print(f"Eval accuracy (full plate): {acc:.4f} | Char accuracy: {acc_char:.4f}")
    return acc, acc_char



def resize_images(input_dir, output_dir, new_size):
    '''
    for YOLO finetuning, I resize all images to a fixed size
    '''
    for filename in os.listdir(input_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  #didn't check all the 11k imgs, want to be sure
            try:
                img_path = os.path.join(input_dir, filename)
                with Image.open(img_path) as img:
                    # Resize the image (using LANCZOS instead of deprecated ANTIALIAS)
                    img_resized = img.resize(new_size, Image.LANCZOS)   #applying a nice filter to sharpen the image

                    # Save the resized image to the output directory
                    output_path = os.path.join(output_dir, filename)
                    img_resized.save(output_path)

                    print(f'Resized {filename} and saved to {output_dir}')
            except Exception as e:
                print(f'Error resizing {filename}: {e}')


## CRNN CLASS
class CRNN(nn.Module):
    def __init__(self, num_classes, input_channels=3):  # Added parameter for flexibility
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, 1, 1), nn.ReLU(),  # Changed from 1 to input_channels
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None))  # (height=1, width stays)
        )
        self.rnn = nn.LSTM(256, 128, num_layers=2, bidirectional=True)
        self.fc = nn.Linear(128*2, num_classes)  # bidirectional

    def forward(self, x):
        x = self.cnn(x)  # [B, C, 1, W]
        x = x.squeeze(2) # [B, C, W]
        x = x.permute(2, 0, 1)  # [W, B, C]
        x, _ = self.rnn(x)
        x = self.fc(x)  # [W, B, num_classes]
        return x  # output for CTC: [seq_len, batch, num_classes]



## dataset for the txt files for the YOLO finetuning
class LicensePlateDataset:
    def __init__(self, img_dir, output_dir, original_size=(720, 1160), target_size=(640, 640)):
        self.img_dir = img_dir
        self.output_dir = output_dir
        self.image_names = os.listdir(img_dir)
        self.original_size = original_size  # (width, height) of original images
        self.target_size = target_size      # (width, height) of resized images
        
        # Calculate scaling factors
        self.scale_x = target_size[0] / original_size[0]  # 640 / 720
        self.scale_y = target_size[1] / original_size[1]  # 640 / 1160

    def get_bbox_coords_YOLO_format(self, filename):
        # Parse original coordinates from filename
        fields = filename.split('-')
        
       
        vertices = fields[3].split('_')
        points = [tuple(map(int, p.split("&"))) for p in vertices]
        
        # Get bounding box from vertices
        xs = [x for x, y in points]
        ys = [y for x, y in points]
        
        x_min_orig, x_max_orig = min(xs), max(xs)
        y_min_orig, y_max_orig = min(ys), max(ys)
        
        # Scale coordinates to resized image dimensions, the coordinates in the filename are referred 
        # to original size, but I resized the images for the finetuning. So I gota scale the coordinates
        x_min_scaled = x_min_orig * self.scale_x
        x_max_scaled = x_max_orig * self.scale_x
        y_min_scaled = y_min_orig * self.scale_y
        y_max_scaled = y_max_orig * self.scale_y
        
        # Calculate center and dimensions in scaled image
        x_center_scaled = (x_min_scaled + x_max_scaled) / 2
        y_center_scaled = (y_min_scaled + y_max_scaled) / 2
        width_scaled = x_max_scaled - x_min_scaled
        height_scaled = y_max_scaled - y_min_scaled
        
        # Normalize to [0, 1] range for YOLO format
        x_center_norm = x_center_scaled / self.target_size[0]
        y_center_norm = y_center_scaled / self.target_size[1]
        width_norm = width_scaled / self.target_size[0]
        height_norm = height_scaled / self.target_size[1]
        
        # Clamp values to [0, 1] to handle any edge cases
        x_center_norm = max(0, min(1, x_center_norm))
        y_center_norm = max(0, min(1, y_center_norm))
        width_norm = max(0, min(1, width_norm))
        height_norm = max(0, min(1, height_norm))
        
        return (x_center_norm, y_center_norm, width_norm, height_norm)

    def create_annotation_files(self):
        for img_name in self.image_names:
            x_center, y_center, yolo_width, yolo_height = self.get_bbox_coords_YOLO_format(img_name)
            class_id = 0  # license plate is the only class

            annotation_file = os.path.join(self.output_dir, f"{os.path.splitext(img_name)[0]}.txt")
            
            with open(annotation_file, 'w') as f:
                f.write(f"{class_id} {x_center} {y_center} {yolo_width} {yolo_height}\n")

    def visualize_bbox(self, filename, save_path=None):
        """
        Visualize the bounding box on the resized image to verify correctness
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        # Load the resized image
        img_path = os.path.join(self.img_dir, filename)
        image = Image.open(img_path)
        
        # Get YOLO coordinates
        x_center, y_center, width, height = self.get_bbox_coords_YOLO_format(filename)
        
        # Convert YOLO format back to pixel coordinates for visualization
        img_width, img_height = image.size
        x_center_px = x_center * img_width
        y_center_px = y_center * img_height
        width_px = width * img_width
        height_px = height * img_height
        
        # Calculate top-left corner for rectangle
        x_min = x_center_px - width_px / 2
        y_min = y_center_px - height_px / 2
        
        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.imshow(image)
        
        # Add bounding box
        rect = patches.Rectangle((x_min, y_min), width_px, height_px,
                               linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        
        # Add center point
        ax.plot(x_center_px, y_center_px, 'ro', markersize=8)
        
        ax.set_title(f'Bounding Box Visualization\n{filename}')
        ax.set_xlabel(f'YOLO coords: ({x_center:.3f}, {y_center:.3f}, {width:.3f}, {height:.3f})')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        else:
            plt.show()
        plt.close()


    





## DATASET CLASS
class CarPlateDataset(Dataset):

    def __init__(self, img_dir, transform=None, cropped = False):
        self.img_dir = img_dir
        self.transform = transform
        self.image_names = os.listdir(img_dir)
        self.cropped = cropped


    def __len__(self):
        return len(self.image_names)

    def parse_filename(self, filename):
        fields = filename.split('-')
        area = float(fields[0]) / 100  #filename encodes the area in percentage (ratio plate-no plate area), so divising by 100 gives me a 0-1 range
        tilt_degree = fields[1].split('_')
        h_tilt = int(tilt_degree[0])    #horizontal tilt degree
        v_tilt = int(tilt_degree[1])    #vertical tilt degree
        tilt_list = np.array([h_tilt, v_tilt], dtype=np.float32)


        bbox_coords = fields[2].split('_')  #bounding box coordinates
        leftUp_bbox = bbox_coords[0].split('&')
        leftUp_bbox_x = int(leftUp_bbox[0])
        leftUp_bbox_y = int(leftUp_bbox[1])
        rightBottom_bbox = bbox_coords[1].split('&')
        rightDown_bbox_x = int(rightBottom_bbox[0])
        rightDown_bbox_y = int(rightBottom_bbox[1])
        bbox_coords_list = np.array([(leftUp_bbox_x, leftUp_bbox_y),
                                    (rightDown_bbox_x, rightDown_bbox_y)], dtype=np.float32)

        vertices = fields[3].split('_')  #vertices of the plate
        right_bottom_vertex = vertices[0].split('&')
        right_bottom_vertex_x = int(right_bottom_vertex[0])
        right_bottom_vertex_y = int(right_bottom_vertex[1])
        left_bottom_vertex = vertices[1].split('&')
        left_bottom_vertex_x = int(left_bottom_vertex[0])
        left_bottom_vertex_y = int(left_bottom_vertex[1])
        left_up_vertex = vertices[2].split('&')
        left_up_vertex_x = int(left_up_vertex[0])
        left_up_vertex_y = int(left_up_vertex[1])
        right_up_vertex = vertices[3].split('&')
        right_up_vertex_x = int(right_up_vertex[0])
        right_up_vertex_y = int(right_up_vertex[1])
 
        vertices_list = np.array([(left_bottom_vertex_x, left_bottom_vertex_y),
                                (right_bottom_vertex_x, right_bottom_vertex_y),
                                (right_up_vertex_x, right_up_vertex_y),
                                (left_up_vertex_x, left_up_vertex_y)], dtype=np.float32)
        



        provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
        alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']
        ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

        
        text=str(fields[4])
        indices=text.split("_")
        province_character=provinces[int(indices[0])]
        alphabet_character=alphabet[int(indices[1])]
        ads_charachters=[ads[int(i)] for i in indices[2:]]
        plate_text=province_character+alphabet_character+"".join(ads_charachters)
        #return plate_text

        brightness = int(fields[5])
        #blurriness = int(fields[6].strip('.jpg'))  # Remove .jpg, it's end of filename
        #it gave problems, try with this
        blurriness_str = fields[6].replace('.jpg', '')
        match = re.match(r'\d+', blurriness_str)
        if match:
            blurriness = int(match.group())
        else:
            print(f"[WARNING] File '{filename}': blurriness non standard '{fields[6]}', imposto a 0.")
            blurriness = 0

        # Convert license plate text to indices for CTC training
        lp_indexes = [char2idx[c] for c in plate_text if c in char2idx]
        
        return {
            'area': area,
            'tilt': tilt_list,
            'bbox_coords': bbox_coords_list,
            'vertices': vertices_list,
            'lp': plate_text,
            'lp_indexes': lp_indexes,
            'brightness': brightness,
            'blurriness': blurriness,
        }

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # Load the image
        image = Image.open(img_path)


        # Parse the filename to get the associated metadata
        metadata = self.parse_filename(img_name)

        if self.cropped:    #I use this dataset for both baselines, so I check if I need to skip detection part and use dataset bbox.
            #I can use the crop method of PIL, that crops the image using coords in this way: (left, upper, right, lower)
            '''
            left is the x-coordinate of the left edge.

            upper is the y-coordinate of the top edge.

            right is the x-coordinate of the right edge.

            lower is the y-coordinate of the bottom edge.
            seen on the online odcs of pillow
            '''
            bbox_coords = metadata['bbox_coords']
            
            left = int(bbox_coords[0][0])   # x-coordinate of the left edge
            upper = int(bbox_coords[0][1])  # y-coordinate of the top edge
            right = int(bbox_coords[1][0])  # x-coordinate of the right edge
            lower = int(bbox_coords[1][1])  # y-coordinate of the bottom edge

            image = image.crop((left, upper, right, lower))

 
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(metadata['lp_indexes'], dtype=torch.long)  # Return the image and the license plate indexes as a tensor, for the CNN to elaborate

    #I included this method in the above one, with the if cropped check. I dunno if i'm gonna need it anymore. COMMENTED FOR NOW
    '''def get_cropped_plate(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path)

        #now I crop the image using the bbox coords that I have in the metadata
        metadata = self.parse_filename(img_name)
        #I can use the crop method of PIL, that crops the image using coords in this way: (left, upper, right, lower)
        
        #left is the x-coordinate of the left edge.

        #upper is the y-coordinate of the top edge.

        #right is the x-coordinate of the right edge.

        #lower is the y-coordinate of the bottom edge.
        #seen on the online odcs of pillow
        
        bboox_coords = metadata['bbox_coords']
        
        left = int(bboox_coords[0][0])   # x-coordinate of the left edge
        upper = int(bboox_coords[0][1])  # y-coordinate of the top edge
        right = int(bboox_coords[1][0])  # x-coordinate of the right edge
        lower = int(bboox_coords[1][1])  # y-coordinate of the bottom edge

        cropped_lp = image.crop((left, upper, right, lower))

        if self.transform:
            cropped_lp = self.transform(cropped_lp)

        return cropped_lp, metadata['lp']'''
    

    
# BOUNDING BOX FUNCTION 

def get_bounding_box(file):
    numbers=file.split("-")
    values=numbers[3]
    values_v2=values.split("&")
    values_v3=[]
    for i in range(len(values_v2)):
        if "_" in values_v2[i]:
            values_v3.append(values_v2[i].split("_"))
    t=[values_v2[0],values_v3[0],values_v3[1],values_v3[2],values_v2[-1]]
    final_values = [int(x) for item in t for x in (item if isinstance(item, list) else [item])]
    x_coords=[final_values[0],final_values[2],final_values[4],final_values[6]]
    y_coords=[final_values[1],final_values[3],final_values[5],final_values[7]]
    x_min = min(x_coords)
    y_min = min(y_coords)
    x_max = max(x_coords)
    y_max = max(y_coords)
    
    return [float(x_min), float(y_min), float(x_max), float(y_max)]

# INTERSECTION OVER UNION FUNCTION

def compute_IoU(box1, box2):

    box1=box1.squeeze()
    box2=box2.squeeze()

    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    area_of_intersection = max(0, xB - xA) * max(0, yB - yA)

    area_box1 = (box1[2] - box1[0]) * (box1[3]- box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    IoU = area_of_intersection / float(area_box1 + area_box2 - area_of_intersection)

    return IoU

# CROP FUNCTION WITH PREDICTED BOUNDING BOX

def crop_image_with_RCNN(file):
    
    image = Image.open(file).convert("RGB")
    transform = T.ToTensor()
    img_tensor = transform(image).unsqueeze(0).to(device) 
    with torch.no_grad():
        prediction = model(img_tensor)[0]
        best_bb=prediction['boxes'][0].to(device)
        best_bb=best_bb.int()
        cropped_image = img_tensor[0, :, best_bb[1]:best_bb[3], best_bb[0]:best_bb[2]]
    return cropped_image

def crop_folder_with_RCNN(folder_path):
    cropped_folder = []
    files = os.listdir(folder_path)
    for file in files:
        full_path = os.path.join(folder_path, file)
        cropped_image = crop_image_with_RCNN(full_path)
        cropped_folder.append(cropped_image)
    return cropped_folder

# DATASET   

class CCPD_Dataset(torch.utils.data.Dataset):
    
    def __init__(self, path, transforms=None):
        self.path = path
        self.transforms = transforms
        self.folder = os.listdir(path)
        self.images = [f for f in self.folder if f.endswith(".jpg")]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        file = self.images[idx]
        full_path = os.path.join(self.path, file)
        image = Image.open(full_path).convert("RGB")

        bbox = get_bounding_box(file)
        tensor_bbox = torch.tensor([bbox], dtype=torch.float32)
        label = torch.tensor([1], dtype=torch.int64)  

        target = {"boxes": tensor_bbox, "labels": label}

        if target["boxes"].shape[0] != target["labels"].shape[0]:
            raise ValueError(f"Mismatch in number of boxes and labels for file: {file}")

        if self.transforms:
            image = self.transforms(image)

        return image, target
    
def collate_fn(batch):
    return tuple(zip(*batch))

# LOADING THE MODEL

def load_Fasterrcnn(device):
    model = fasterrcnn_resnet50_fpn(num_classes=2)  
    model.load_state_dict(torch.load('model_weights/best_frcnn_model.pth'))
    model.to(device)
    model.eval()