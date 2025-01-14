import random
import numpy as np
from util import Dict
from util import STATE_STEP_DIM, STATE_STOPPED_DIM
from dataset import LoadImagesAndLabelsRAWReplay_target, LoadImagesAndLabelsRAWReplay, LoadImagesAndLabelsNormalizeReplay
import torch
from isp_blocks import ISPBlocks
from agents.states import get_init_state_supp


def create_input_tensor(batch, objective='yolo'):
    im_list, label_list, path_list, shapes_list, states_list = batch
    if objective != 'l2':
        for i, lb in enumerate(label_list):
            lb[:, 0] = i  # add target image index for build_targets()
    if objective == 'l2':
        return torch.from_numpy(np.stack(im_list, 0)), \
            torch.from_numpy(np.stack(label_list, 0)), path_list, shapes_list, \
            torch.from_numpy(np.stack(states_list, 0))
    return torch.from_numpy(np.stack(im_list, 0)), \
           torch.from_numpy(np.concatenate(label_list, 0)), path_list, shapes_list, \
           torch.from_numpy(np.stack(states_list, 0))


def get_noise(batch_size, z_type="uniform", z_dim=27):
    if z_type == 'normal':
        return np.random.normal(0, 1, [batch_size, z_dim]).astype(np.float32)
    elif z_type == 'uniform':
        return np.random.uniform(0, 1, [batch_size, z_dim]).astype(np.float32)
    else:
        assert False, 'Unknown noise type: %s' % z_type


def get_initial_states(batch_size, num_state_dim, filters_number):
    states = np.zeros(shape=(batch_size, num_state_dim), dtype=np.float32)
    for k in range(batch_size):
        for i in range(len(filters_number)):
            # states[k, -(i + 1)] = 1 if random.random() < self.cfg.filter_dropout_keep_prob else 0
            # Used or not?
            # Initially nothing has been used
            states[k, -(i + 1)] = 0
    return states

'''
in new settings:
state will have ->
- original input
- current step
- the landscape of steps (how many rows? depends on ISP pipeline)
- stop signal, when to stop, max iter
- past param (or just current param -- to be operated on)
- last time reward value?

state can be determined on ->
- isp pipeline
- 

each time ->
img -> para, retouch, reward, delta, selected id 

we can be selective on model inp state
but this state_supp dict must have all
-> we could unified state_supp Dict
-> different state to inp_state converter function
'''


class ReplayMemory:
    def __init__(self,
                 cfg,
                 load,
                 path,
                 imgsz,
                 batch_size,
                 stride,
                 single_cls=False,
                 hyp=None,
                 augment=False,
                 cache=False,
                 pad=0.0,
                 rect=False,
                 image_weights=False,
                 prefix='',
                 limit=-1,
                 data_name='coco',
                 add_noise=False,
                 brightness_range=None,
                 noise_level=None,
                 use_linear=False,
                 pool_size=None,
                 use_dataset_len_as_pool_size=False,
                 isp_blocks:ISPBlocks=None,
                 mem_traj_keep_len=None):
        self.cfg = cfg
        if data_name == "isp_diff":
            self.dataset = LoadImagesAndLabelsRAWReplay_target(
                path,
                imgsz,
                batch_size,
                augment=augment,  # augmentation
                hyp=hyp,  # hyperparameters
                rect=rect,  # rectangular batches
            )
        elif data_name == "coco":
            self.dataset = LoadImagesAndLabelsRAWReplay(
                path,
                imgsz,
                batch_size,
                augment=augment,  # augmentation
                hyp=hyp,  # hyperparameters
                rect=rect,  # rectangular batches
                cache_images=cache,
                single_cls=single_cls,
                stride=int(stride),
                pad=pad,
                image_weights=image_weights,
                prefix=prefix,
                limit=limit,
                add_noise=add_noise,
                brightness_range=brightness_range,
                noise_level=noise_level,
                use_linear=use_linear,
            )
        elif data_name in ("lod", "oprd"):
            self.dataset = LoadImagesAndLabelsNormalizeReplay(
                path,
                imgsz,
                batch_size,
                augment=augment,  # augmentation
                hyp=hyp,  # hyperparameters
                rect=rect,  # rectangular batches
                cache_images=cache,
                single_cls=single_cls,
                stride=int(stride),
                pad=pad,
                image_weights=image_weights,
                prefix=prefix,
                limit=limit,
            )
        else:
            raise ValueError("ReplayMemory input data_name error!")
        # The images with labels of #operations applied
        self.image_pool = []
        self.target_pool_size = len(self.dataset) if use_dataset_len_as_pool_size \
            else (cfg.replay_memory_size if (pool_size is None) else pool_size)
        self.mem_traj_keep_len = mem_traj_keep_len if mem_traj_keep_len is not None \
            else cfg.maximum_trajectory_length
        self.fake_output = None
        self.batch_size = batch_size
        self.isp_blocks = isp_blocks
        if load:
            print("start loading replay mem")
            self.load()
            print("done loading replay mem")

    def load(self):
        self.fill_pool()

    def get_initial_states(self, batch_size):
        states = np.zeros(shape=(batch_size, self.cfg.num_state_dim), dtype=np.float32)
        for k in range(batch_size):
            for i in range(len(self.cfg.filters)):
                # states[k, -(i + 1)] = 1 if random.random() < self.cfg.filter_dropout_keep_prob else 0
                # Used or not?
                # Initially nothing has been used
                states[k, -(i + 1)] = 0
        return states

    def fill_pool(self):
        while len(self.image_pool) < self.target_pool_size:
            im_list, label_list, path_list, shapes_list = self.dataset.get_next_batch(self.batch_size)
            for i in range(len(im_list)):
                self.image_pool.append(Dict(
                    im=im_list[i],
                    label=label_list[i],
                    path=path_list[i],
                    shape=shapes_list[i],
                    state=self.get_initial_states(1)[0],
                    state_supp=(
                        get_init_state_supp(img=im_list[i], isp=self.isp_blocks, max_step=self.cfg.test_steps)
                        if self.isp_blocks is not None else None
                    )
                ))
        self.image_pool = self.image_pool[:self.target_pool_size]
        assert len(self.image_pool) == self.target_pool_size, '%d, %d' % (
            len(self.image_pool), self.target_pool_size)

    def get_next_RAW(self, batch_size):
        im_list, label_list, path_list, shapes_list = self.dataset.get_next_batch(batch_size)
        pool = []
        for i in range(len(im_list)):
            pool.append(Dict(
                im=im_list[i],
                label=label_list[i],
                path=path_list[i],
                shape=shapes_list[i],
                state=self.get_initial_states(1)[0],
                state_supp=get_init_state_supp(
                    img=im_list[i],
                    isp=self.isp_blocks,
                    max_step=self.cfg.test_steps
                )
            ))
        return self.records_to_images_and_states(pool)

    def get_feed_dict_and_states(self, batch_size):
        images, labels, paths, shapes, states, states_supp = self.get_next_fake_batch(batch_size)
        z = self.get_noise(batch_size)
        data = {
            "im": images,   # list
            "label": labels,  # list
            "path": paths,  # list -> not important
            "shape": shapes,  # list -> not actually used
            "state": states,  # list
            "state_supp": states_supp,  # list
            "z": z   # numpy
        }
        #
        return data

    # Not actually used.
    def get_noise(self, batch_size):
        if self.cfg.z_type == 'normal':
            return np.random.normal(0, 1, [batch_size, self.cfg.z_dim]).astype(np.float32)
        elif self.cfg.z_type == 'uniform':
            return np.random.uniform(0, 1, [batch_size, self.cfg.z_dim]).astype(np.float32)
        else:
            assert False, 'Unknown noise type: %s' % self.cfg.z_type

    # Note, we add finished images since the discriminator needs them for training.
    def replace_memory(self, new_images):
        random.shuffle(self.image_pool)
        # Insert only PART of new images
        # TODO: if inserted image is never used, why occupy space
        for r in new_images:
            if r.state[STATE_STEP_DIM] < self.mem_traj_keep_len or random.random(
            ) < self.cfg.over_length_keep_prob:
                self.image_pool.append(r)
        # ... and add some brand-new RAW images
        self.fill_pool()
        random.shuffle(self.image_pool)

    # For supervised learning case, images should be [batch size, 2, channels, size, size]
    @staticmethod
    def records_to_images_and_states(batch):
        im_list = [x['im'] for x in batch]
        label_list = [x['label'] for x in batch]
        path_list = [x['path'] for x in batch]
        shapes_list = [x['shape'] for x in batch]
        states_list = [x['state'] for x in batch]
        states_supp_list = [x['state_supp'] for x in batch]
        return im_list, label_list, path_list, shapes_list, states_list, states_supp_list
        # for i, lb in enumerate(label_list):
        #     lb[:, 0] = i  # add target image index for build_targets()
        # return np.stack(im_list, 0), np.concatenate(label_list, 0), path_list, shapes_list,
        # np.stack(states_list, axis=0)

    @staticmethod
    def images_and_states_to_records(images, labels, paths, shapes, states, states_supp=None):
        assert len(images) == len(states)
        if states_supp is not None:
            assert len(states) == len(states_supp)
        else:
            states_supp = [None for _ in range(len(images))]
        records = []
        for i in range(len(images)):
            records.append(Dict(
                im=images[i],
                label=labels[i],
                path=paths[i],
                shape=shapes[i],
                state=states[i],
                state_supp=states_supp[i]
            ))
        return records

    def get_next_fake_batch(self, batch_size):
        # print('get_next')
        random.shuffle(self.image_pool)
        assert batch_size <= len(self.image_pool)
        batch = []
        while len(batch) < batch_size:
            if len(self.image_pool) == 0:
                self.fill_pool()
            record = self.image_pool[0]
            self.image_pool = self.image_pool[1:]
            if record.state[STATE_STOPPED_DIM] != 1:
                # TODO We avoid adding any finished images here.
                batch.append(record)
        return self.records_to_images_and_states(batch)

    def debug(self):
        tot_trajectory = 0
        for r in self.image_pool:
            tot_trajectory += r.state[STATE_STEP_DIM]
        average_trajectory = 1.0 * tot_trajectory / len(self.image_pool)
        print('# Replay memory: size %d, avg. traj. %.2f' % (len(self.image_pool),
                                                             average_trajectory))
        print('#--------------------------------------------')


if __name__ == "__main__":
    import yaml
    from config import cfg

    cfg.replay_memory_size = 2

    train_path = "/mnt/data/sail_3090/wujiarui/data/distilled_jpg_pair/val"
    hyp = 'yolov3/data/hyps/hyp.scratch-low.yaml'
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    memory = ReplayMemory(cfg, True, train_path, 512, 1, 32, False, None,
                          rect=False, prefix='train: ', limit=1)
    # test get batch data
    feed_dict = memory.get_feed_dict_and_states(1)
    print(feed_dict.keys())
    for k, v in feed_dict.items():
        print(k, type(v))
        if k == "path":
            print(v)

    feed_dict['path'][0] = "1"
    print(feed_dict['path'])
    # test update value
    memory.replace_memory(memory.images_and_states_to_records(feed_dict['im'], feed_dict['label'], feed_dict['path'],
                                                              feed_dict['shape'], feed_dict['state']))
    feed_dict = memory.get_feed_dict_and_states(1)
    print(feed_dict.keys())
    for k, v in feed_dict.items():
        print(k, type(v))
        if k == "path":
            print(v)
    feed_dict = memory.get_feed_dict_and_states(1)
    print(feed_dict.keys())
    for k, v in feed_dict.items():
        print(k, type(v))
        if k == "path":
            print(v)