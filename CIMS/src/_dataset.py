from pathlib import Path
import numpy as np
import h5py
from torch.utils.data import Dataset
from mmcv.transforms import Compose

from mmengine.registry import DATASETS

@DATASETS.register_module()
class MultiChannelDataset(Dataset):

    def __init__(self,
                h5_file_path,
                patch_size,
                shuffle: bool = False,
                markers_to_use= None,
                in_memory=False,
                image_key='IMAGES',
                marker_key='MARKERS',
                pipeline=None,
                additional_keys=[],
                split=[0.7, 0.1, 0.2],
                used_split='training',
                mask_image=False, 
                classes_to_ignore=None,
                **kwargs):
        
        super().__init__()
           
        # Ensure the HDF5 file exists
        assert Path(h5_file_path).exists(), f"Provided path to h5 file does not exist: {h5_file_path}"
        
        self.image_key = image_key
        self.marker_key = marker_key
        self.additional_keys = additional_keys
        self.split = split
        self.used_split = used_split
        self.mask_image = mask_image
        self.classes_to_ignore = classes_to_ignore
        self.markers_to_use = markers_to_use
        self.shuffle = shuffle
        
        self.h5_file_path = h5_file_path
        self.h5_file = h5py.File(h5_file_path, 'r')
        self.indexer = self.h5_file['INDEXER'][:]
        annotations = self.h5_file['ANNOTATIONS'][()].astype(str) if 'ANNOTATIONS' in self.h5_file.keys() else None
        
        self.annotations = annotations
        if self.classes_to_ignore is not None and annotations is not None:
            idxs = set(list(range(len(self.indexer))))
            ignored_idxs = []
            for ignored_class in self.classes_to_ignore:
                class_idxs = np.where(self.annotations == ignored_class)[0]
                ignored_idxs.extend(class_idxs)
                print(f'Removing {len(class_idxs)} instances of class:"{ignored_class}"')
            self.row_indexer = np.array(list(idxs - set(ignored_idxs)))
        else:
            self.row_indexer = np.arange(len(self.indexer))
            
        if self.annotations is not None: 
            print(f'Dataset contains annotations')
            
        print(f'Dataset contains {len(self.indexer)} cells.')
        
        
        if in_memory:
            print('Start loading images to memory')
            self.image_data = self.h5_file[image_key][:]
            if 'MASKS' in self.h5_file.keys():
                self.mask_data = self.h5_file['MASKS'][:]
            else:
                self.mask_data = None
            print('Finished loading images to memory')
        else:
            self.image_data = self.h5_file[image_key]
            if 'MASKS' in self.h5_file.keys():
                self.mask_data = self.h5_file['MASKS']
            else:
                self.mask_data = None
                
        self.marker_names = self.h5_file[marker_key][:].astype(str)
        print(f"All Markers: {self.marker_names}")
        
        # Set patch size and calculate half patch size
        self.patch_size = patch_size
        self.half_patch_size = self.hpsz = self.patch_size // 2
        assert self.patch_size % 2 == 0, "patch_size needs to be divisible by 2"
        
        # Retrieve dimensions from the first image in the IMC dataset
        self.N, self.W, self.H, self.C = self.image_data.shape
        print(f"Image data has original shape: {self.W} x {self.H} x {self.C}")
        
        # self.channels_to_skip = channels_to_skip
        if markers_to_use is not None:
            self.markers_to_use = markers_to_use
            
            channel_idxs = []
            for marker in self.markers_to_use:
                if marker not in self.marker_names:
                    raise ValueError(f'{marker} is not in marker names list!')
                channel_idxs.append(np.where(self.marker_names == marker)[0][0])
            self.marker_names = self.marker_names[channel_idxs]
            self.used_channels = np.array(channel_idxs)
            
        else:
            self.used_channels = np.arange(len(self.marker_names))
            
        print(f"Used Markers: {self.marker_names}")
            
        self.channel2idx = {channel: i for i, channel in enumerate(self.marker_names)}
        self.idx2channel = {i: channel for i, channel in enumerate(self.marker_names)}

        # Set additional parameters
        # self.exclude_edges = exclude_edges    
                             
        # Get indexer and setup shuffling
        if self.shuffle:
            np.random.shuffle(self.row_indexer)

        if pipeline is not None:
            self.pipeline = Compose(pipeline)
        else:
            self.pipeline = lambda x: x
            
        if self.split is not None:
            
            if used_split == 'all':
                self.row_indexer = self.row_indexer
                print(f'Using {self.used_split} split!')

            else:
            
                assert len(self.split) == 3, f'Please provide a split of percentages for train, val, test'
                assert sum(self.split) == 1, f'Please provide the splits as percentages that sum to 1'
                
                self.random_state = kwargs.get('random_state', 42)
                np.random.seed(self.random_state)
                np.random.shuffle(self.row_indexer)
            
                n_train = int(len(self.row_indexer) * self.split[0])
                n_val = int(len(self.row_indexer) * self.split[1])
                n_test = len(self.row_indexer) - n_train - n_val
                
                print(n_train, n_val, n_test)
                
                train_idxs = np.sort(self.row_indexer[:n_train])
                val_idxs = np.sort(self.row_indexer[n_train:(n_train+n_val)])
                test_idxs = np.sort(self.row_indexer[(n_train+n_val):])
                
                if used_split == 'training':
                    self.row_indexer = train_idxs
                elif used_split == 'validation':
                    self.row_indexer = val_idxs
                elif used_split == 'test':
                    self.row_indexer = test_idxs
                    
                print(f'Using {self.used_split} split!')
            
        #print(self.__repr__())
            
        # self.row_idxs = self.get_row_idxs()

        # self.shuffle = shuffle
        # if self.shuffle:
        #     np.random.shuffle(self.row_idxs)
        #     print("Shuffling data order")

        # if pipeline is not None:
        #     self.pipeline = Compose(pipeline)
        # else:
        #     self.pipeline = lambda x: x
         
            
    def __len__(self):
        return len(self.row_indexer)


    def __getitem__(self, idx, eval=False, ommit_pipeline=False):
        
        dataset_idx = idx
        idx = self.row_indexer[idx]
        
        stack_idx, x, y = self.indexer[idx].astype(int)
        curr_patch = self.get_patch(self.image_data, stack_idx, y, x, self.hpsz)[..., self.used_channels]
        if self.mask_data is not None:
            curr_mask = self.get_patch(self.mask_data, stack_idx, y, x, self.hpsz)
        else:
            curr_mask = []
            
        if self.mask_image:
            mask_value = curr_mask[curr_mask.shape[0]//2, curr_mask.shape[1]//2]
            curr_mask[curr_mask != mask_value] = 0
            curr_mask[curr_mask == mask_value] = 1
            curr_patch = curr_patch * curr_mask[..., None]
        
        pipeline_dict = {
            'img': curr_patch.astype(np.float32),
            'batch': stack_idx,
            'masks': curr_mask,
            'idx': idx,
            'dataset_idx': dataset_idx,
            '(x,y)': (x,y),
            'annotation': self.annotations[idx]
        }
        

        for key in self.additional_keys:
            data = self.h5_file[key][idx]
            data = data if type(data) != bytes else data.decode("utf-8")
            pipeline_dict[key] = data
        
        return  self.pipeline(pipeline_dict)
    

    def get_patch(self, image_stack, im_idx, y, x, hpsz):

        x_min, x_max = max(x - hpsz, 0), min(x + hpsz, self.H)
        pad_x_min = max(0, hpsz - x)
        pad_x_max = max(0, (x + hpsz) - self.H)

        y_min, y_max = max(y - hpsz, 0), min(y + hpsz, self.W)
        pad_y_min = max(0, hpsz - y)
        pad_y_max = max(0, (y + hpsz) - self.W)
        
        if pad_x_min + pad_x_max + pad_y_min + pad_y_max == 0:
            img = image_stack[im_idx, y_min:y_max, x_min:x_max]
        else:
            img = image_stack[im_idx, y_min:y_max, x_min:x_max]
            if img.ndim == 2:
                padding = ((pad_y_min, pad_y_max), (pad_x_min, pad_x_max))
            elif img.ndim == 3:
                padding = ((pad_y_min, pad_y_max), (pad_x_min, pad_x_max), (0,0))
                
            img = np.pad(img, padding, mode='constant', constant_values=0)

        if img.shape[:2] != (hpsz*2, hpsz*2):
            
            print('ERROR')
            # print(img.shape)
            # print(x_min, x_max, y_min, y_max, pad_x_min, pad_x_max, pad_y_min, pad_y_max)
            # print(x_min - x_max, y_min - y_max, pad_x_min, pad_x_max, pad_y_min, pad_y_max)
            
            # if pad_x_min + pad_x_max + pad_y_min + pad_y_max == 0:
            #     img = image_stack[im_idx, y_min:y_max, x_min:x_max, :]
            # else:
            #     img = np.pad(image_stack[im_idx, y_min:y_max, x_min:x_max, :], ((pad_y_min, pad_y_max), (pad_x_min, pad_x_max), (0,0)), mode='constant', constant_values=0)

        return img
    
    def __repr__(self):
        
        self.curr_annotations = self.annotations[self.row_indexer]
        
        prnt = ''
        prnt += f'H5-File: {self.h5_file_path}\n'
        prnt += f'Using {self.used_split} split\n\n'
        prnt += f'Dataset Summary:\n'
        
        celltypes, counts = np.unique(self.curr_annotations, return_counts=True)
        total = counts.sum()
        sorted_idx = np.argsort(counts)[::-1]

        prnt += f"{'Celltype':<25}{'Count':>10}{'Percent':>12}\n"
        prnt += "-" * 47 + "\n"

        for idx in sorted_idx:
            ct = celltypes[idx]
            cnt = counts[idx]
            pct = cnt / total * 100
            prnt += f"{ct:<25}{cnt:>10}{pct:>11.2f}%\n"

        prnt += "-" * 47 + "\n"
        prnt += f"{'Total':<25}{total:>10}{100:>11.2f}%\n"
        
        return prnt
                
        # orig_idx = idx
        
        # idx = self.row_idxs[idx] # if not shuffle: eg orig_idx:134 -> idx:134 | if shuffle eg orig_idx:134 -> idx:829
        
        # img_idx, x, y, _, _ = self.h5_file["INDEXER"][idx].astype(int)
                  
        # # Extract and pad patches and masks
        # current_patch = self.extract_and_pad(img_idx, x, y)

        # current_patch = current_patch[..., self.used_channels]

        # pipeline_dict = {
        #     'img': current_patch.astype(np.float32),
        #     'batch': img_idx,
        #     'idx': idx,
        #     'masks': [],
        #     '(x,y)': (x,y)
        # }
        
  
        # if ommit_pipeline:
        #     return pipeline_dict
                    
        # return self.pipeline(pipeline_dict)

    # def extract_and_pad(self, img_idx, x, y):
        
    #     x_min, x_max = max(y - self.hpsz, 0), min(y + self.hpsz, self.W)
    #     pad_x_min, pad_x_max = max(0, self.hpsz - y), max(0, (y + self.hpsz) - self.W)

    #     y_min, y_max = max(x - self.hpsz, 0), min(x + self.hpsz, self.H)
    #     pad_y_min, pad_y_max = max(0, self.hpsz - x), max(0, (x + self.hpsz) - self.H)

    #     current_patch = self.h5_file[f"IMAGES"][img_idx, x_min:x_max, y_min:y_max]

    #     current_patch = np.pad(
    #         current_patch,
    #         ((pad_x_min, pad_x_max), (pad_y_min, pad_y_max), (0, 0)),
    #         mode='constant',
    #         constant_values=0
    #     )

    #     return current_patch
    

    # def get_row_idxs(self):
    #     # Load the indexer data
    #     # indexer [batch_idx, y_coord, x_coord] 
        
    #     h5_idxs = self.h5_file["INDEXER"][()].astype(int)
        
    #     total_cells = len(h5_idxs)
        
    #     to_keep_sample = np.arange(total_cells)
       
    #     # If edge exclusion is not needed, return the full indexer
    #     if not self.#exclude_edges:
    #         return to_keep_sample

    #     # Get the image dimensions per row
    #     W_all = h5_idxs[:, 3]
    #     H_all = h5_idxs[:, 4]

    #     # Get the x, y positions and image indices
    #     img_idx_all = h5_idxs[:, 0]
    #     x_all = h5_idxs[:, 1]
    #     y_all = h5_idxs[:, 2]

    #     # Compute the thresholds
    #     W_thresh = W_all - self.hpsz
    #     H_thresh = H_all - self.hpsz

    #     # Apply the full mask vectorized
    #     valid_mask = (
    #         (x_all > self.hpsz) & (x_all < W_thresh) &
    #         (y_all > self.hpsz) & (y_all < H_thresh)
    #     )

    #     # Get the indices that meet the condition
    #     to_keep_edges = np.where(valid_mask)[0]

    #     # Intersect with existing mask
    #     to_keep = np.intersect1d(to_keep_edges, to_keep_sample)

    #     # Logging
    #     num_excluded = len(to_keep_sample) - len(to_keep)
    #     print(f"{len(to_keep)} remaining after excluding {num_excluded:.0f} cells ({num_excluded/len(to_keep_sample):.2%} of all cells) because they are at the edge of the image")
    #             # Return the filtered indexer
    #     return to_keep
    
        
    # def define_used_channels(self):
            
    #     #define the channels that are being used        
    #     if self.markers_to_use is not None:
    #         if not isinstance(self.markers_to_use, list):
    #             self.markers_to_use = [self.markers_to_use]
    #         assert len(self.markers_to_use) <= self.C, "selected more channels to use than available channels"

    #         if isinstance(self.markers_to_use[0], str):
    #             markers_to_use = []
    #             for marker in self.markers_to_use:
    #                 if marker not in self.marker_names:
    #                     raise ValueError(f"{marker} is not in available marker names: {self.marker_names}")
                    
    #                 markers_to_use.append(np.argmax(self.marker_names == marker))

    #             self.markers_to_use = markers_to_use
            
    #         else:
    #             raise TypeError("Please provide channels to use as their name!")
                
    #         used_channels = sorted(self.markers_to_use)
                
    #         print(f"Only using the following channels: {self.markers_to_use}")
    #         print(f"patch dimensions: {self.patch_size} x {self.patch_size} x {len(self.markers_to_use)}")

    #     elif self.channels_to_skip is not None:
    #         if not isinstance(self.channels_to_skip, list):
    #             self.channels_to_skip =[self.channels_to_skip]
    #         assert len(self.channels_to_skip) <= self.C, "selected more channels to skip than available channels"
            
    #         if isinstance(self.channels_to_skip[0], str):
    #             channels_to_skip = []
    #             for marker in self.channels_to_skip:
    #                 if marker not in self.marker_names:
    #                     raise ValueError(f"{marker} is not in available marker names: {self.marker_names}")
                    
    #                 channels_to_skip.append(np.argmax(self.marker_names == marker))

    #             self.channels_to_skip = channels_to_skip
            
    #         else:
    #             raise TypeError("Please provide channels to skip as str!")
            
    #         used_channels = sorted([channel_idx for channel_idx in range(self.C) if not channel_idx in self.channels_to_skip])
                
    #         print(f"Omitting the following channels: {self.channels_to_skip}")
    #         print(f"patch dimensions: {self.patch_size} x {self.patch_size} x {self.C-len(self.channels_to_skip)}")

    #     elif self.channels_to_skip is None and self.markers_to_use is None:
    #         used_channels = list(range(self.C))
    #         print("Using all channels")
    #         print(f"patch dimensions: {self.patch_size} x {self.patch_size} x {self.C}")

    #     channel_names = self.marker_names[used_channels]
        
    #     return used_channels, channel_names
    