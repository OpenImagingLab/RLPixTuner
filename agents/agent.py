# import tensorflow as tf
# import tensorflow.contrib.layers as ly
import cv2
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pdf_sample_layer import pdf_sample
from util import enrich_image_input, enrich_image_input_w_target
from util import STATE_DROPOUT_BEGIN, STATE_REWARD_DIM, STATE_STEP_DIM, STATE_STOPPED_DIM

agent_debug = False

def one_hot(num_class, index):
    label = torch.zeros((num_class, *index.shape), dtype=torch.int64, device=index.device)
    for i in range(num_class):
        label[i, index == i] = 1
    label = label.permute(1, 0)
    return label


class FeatureExtractor(torch.nn.Module):
    def __init__(self, shape=(14, 64, 64), mid_channels=32, output_dim=4096, dropout_prob=0.5):
        """shape: c,h,w"""
        super(FeatureExtractor, self).__init__()
        in_channels = shape[0]
        self.output_dim = output_dim

        min_feature_map_size = 4
        assert output_dim % (min_feature_map_size ** 2) == 0, 'output dim=%d' % output_dim
        size = int(shape[2])
        # print('Agent CNN:')
        # print('    ', shape)
        size = size // 2
        channels = mid_channels
        layers = []
        layers.append(nn.Conv2d(in_channels, channels, kernel_size=4, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(channels))
        layers.append(nn.LeakyReLU(negative_slope=0.2))
        while size > min_feature_map_size:
            in_channels = channels
            if size == min_feature_map_size * 2:
                channels = output_dim // (min_feature_map_size ** 2)
            else:
                channels *= 2
            assert size % 2 == 0
            size = size // 2
            # print(size, in_channels, channels)
            layers.append(nn.Conv2d(in_channels, channels, kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.LeakyReLU(negative_slope=0.2))
        self.layers = nn.Sequential(*layers)
        self.droupout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        # x = x - 0.5
        x = self.layers(x)
        x = torch.reshape(x, [-1, self.output_dim])
        if agent_debug:
            print('reshape after: ', x.shape)
        x = self.droupout(x)
        return x


# Output: float \in [0, 1]
'''
about agent network
- state are actually planes concat to input
- action_sel and action_param are two different net
- is it because this sel - find para order may need filter differential
- we can do in rows -- Ex, WB, CCM, xxx, Ex, WB, CCM ... net will decide stop, thus bypass diff filter
- contrast cemera ccm what so ever inference
-
'''
class Agent(nn.Module):
    def __init__(self, cfg, shape=(16, 64, 64), device='cuda', input_target=False):
        super(Agent, self).__init__()
        if input_target:
            shape = (shape[0] + 3, shape[1], shape[2])
        self.cfg = cfg
        self.feature_extractor = FeatureExtractor(shape=shape, mid_channels=cfg.base_channels,
                                                  output_dim=cfg.feature_extractor_dims,
                                                  dropout_prob=1.0 - cfg.dropout_keep_prob)
        '''param determination policy net'''
        self.filters = []
        for func in self.cfg.filters:
            filter = func(self.cfg, predict=True).to(device)
            self.__setattr__(filter.get_short_name(), filter)
            self.filters.append(filter)

        self.action_selection = FeatureExtractor(shape=shape, mid_channels=cfg.base_channels,
                                                 output_dim=cfg.feature_extractor_dims,
                                                 dropout_prob=1.0 - cfg.dropout_keep_prob)
        '''action selection policy net'''

        self.fc1 = nn.Linear(cfg.feature_extractor_dims, cfg.fc1_size)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.fc2 = nn.Linear(cfg.fc1_size, len(self.filters))
        self.softmax = nn.Softmax()
        self.down_sample = nn.AdaptiveAvgPool2d((shape[1], shape[2]))
        self.input_target = input_target

    def forward(self, inp, progress, high_res=None, specified_filter_id=None):
        train = 1 if self.training else 0
        if self.input_target:
            x, targets, z, states = inp
        else:
            x, z, states = inp

        selection_noise = z[:, 0:1]
        filtered_images = []
        filter_debug_info = []
        high_res_outputs = []

        x_down = self.down_sample(x)
        if self.cfg.shared_feature_extractor:
            if self.input_target:
                targets_down = self.down_sample(targets)
                filter_features = self.feature_extractor(enrich_image_input_w_target(self.cfg, x_down, targets_down, states))
            else:
                filter_features = self.feature_extractor(enrich_image_input(self.cfg, x_down, states))
        else:
            raise ValueError("current just support shared_feature_extractor")
        # filter_features.sum().backward()
        for j, filter in enumerate(self.filters):
            # print('    creating filter:', j, 'name:', str(filter.__class__), 'abbr.', filter.get_short_name())
            # print('      filter_features:', filter_features.shape)
            filtered_image_batch, high_res_output, per_filter_debug_info = filter(x, filter_features, high_res=high_res)
            high_res_outputs.append(high_res_output)
            filtered_images.append(filtered_image_batch)
            filter_debug_info.append(per_filter_debug_info)
            # print('      output:', filtered_image_batch.shape)
            # filtered_image_batch.sum().backward()

        # [batch_size, #filters, H, W, C]
        # for img in filtered_images:
        #     print('img', img.shape)
        filtered_images = torch.stack(filtered_images, dim=1)
        # print('    filtered_images:', filtered_images.shape)

        # filtered_images.sum().backward()
        # action_selection
        if self.input_target:
            selector_features = self.action_selection(enrich_image_input_w_target(self.cfg, x_down, targets_down, states))
        else:
            selector_features = self.action_selection(enrich_image_input(self.cfg, x_down, states))
        # print('    selector features:', selector_features.shape)
        selector_features = self.lrelu(self.fc1(selector_features))

        # print('    selector features:', selector_features.shape)
        pdf = self.softmax(self.fc2(selector_features)) + 1e-37
        # print('    pdf_filter', pdf[:, 1:].shape)

        pdf = pdf * (1 - self.cfg.exploration) + self.cfg.exploration * 1.0 / len(self.filters)
        # pdf = tf.to_float(is_train) * tf.concat([pdf[:, :1], pdf[:, 1:] * states[:, STATE_DROPOUT_BEGIN:]], axis=1) \
        # + (1.0 - tf.to_float(is_train)) * pdf
        pdf = pdf / (torch.sum(pdf, dim=1, keepdim=True) + 1e-30)
        '''
        have pdf now
        '''
        entropy = -pdf * torch.log(pdf)
        entropy = torch.sum(entropy, dim=1)[:, None]
        # print('    pdf:', pdf.shape)
        # print('    entropy:', entropy.shape)
        # print('    selection_noise:', selection_noise.shape)
        random_filter_id = pdf_sample(pdf, selection_noise)
        max_filter_id = torch.argmax(pdf, dim=1).to(torch.int32)

        selected_filter_id = (train * random_filter_id + (1 - train) * max_filter_id).to(torch.int64)
        # print("selected_filter_id", selected_filter_id, random_filter_id, max_filter_id)
        # print('    selected_filter_id:', selected_filter_id)
        '''
        finally we get selected filter
        high res: where im path become useful
        '''

        # selected_filter_id = torch.clip(selected_filter_id, min=0, max=len(self.filters)-1)
        if specified_filter_id is not None:
            assert specified_filter_id in range(0, len(self.filters))
            selected_filter_id[:] = specified_filter_id
            # print(selected_filter_id)

        filter_one_hot = one_hot(len(self.filters), selected_filter_id)

        # print('    filter one_hot', filter_one_hot.shape, filter_one_hot)
        surrogate = torch.sum(filter_one_hot * torch.log(pdf + 1e-10), dim=1, keepdim=True)
        '''
        surrogate: to compute gradient of action selection
        '''

        x = torch.sum(filtered_images * filter_one_hot[:, :, None, None, None], dim=1)
        '''
        final x for output -> x will be computed gradient when optimizing para determination
        '''

        if high_res is not None:
            high_res_outputs = torch.stack(high_res_outputs, dim=1)
            high_res_output = torch.sum(high_res_outputs * filter_one_hot[:, :, None, None, None], dim=1)

        # only the first image will get debug_info
        debug_info = {
            'state': states,
            'selected_filter_id': selected_filter_id[0],
            'filter_debug_info': filter_debug_info,
            'pdf': pdf[0],
            'selected_filter': selected_filter_id,
        }

        # Combined: Three in one 64x64 ?
        #           otherwise returns pdf, detail, mask
        def debugger(debug_info, combined=True):
            size = len(self.cfg.filters)  # 8
            img = None
            images = [None for i in range(3)]
            for i, filter in enumerate(self.filters):
                selected = i == debug_info['selected_filter_id']
                if selected:
                    img = filter.visualize_mask(debug_info['filter_debug_info'][i], (64, 64)) * 0.8
            assert img is not None
            if not combined:
                # Mask
                images[2] = img.copy()
                # reset img
                img = img * 0 + 0.5

            c = 0
            for i, filter in enumerate(self.filters):
                pdf = debug_info['pdf'][i]
                if pdf < 1e-10:
                    continue
                else:
                    c += 1
                selected = i == debug_info['selected_filter_id']
                if selected:
                    filter.visualize_filter(debug_info['filter_debug_info'][i], img)
            if not combined:
                # detail
                images[1] = img.copy()
                # reset img
                img = img * 0 + 0.5
            c = 0
            for i, filter in enumerate(self.filters):
                per_col = (len(self.cfg.filters) + 1) // 2  # 4
                x = c // per_col * 30
                y = size * (c % per_col + 1)
                pdf = debug_info['pdf'][i]
                if pdf < 1e-10:
                    continue
                else:
                    c += 1
                cv2.putText(img, filter.get_short_name(), (x + 6, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.233,
                            (255, 255, 255))
                selected = i == debug_info['selected_filter_id']
                color = 1.0 if selected else 0.3
                width = int(pdf * 20)
                height = 0.35
                corners = [(x + 16, int(y + (1 - height) * size // 2)),
                           (x + 16 + width, int(y + (1 + height) * size // 2))]
                cv2.rectangle(img, (corners[0][0] - 1, corners[0][1] - 1),
                              (corners[1][0] + 1, corners[1][1] + 1), (1, 1, 1), cv2.FILLED)
                cv2.rectangle(img, corners[0], corners[1], (color, 0.3, 0.3), cv2.FILLED)
            if not combined:
                # pdf
                images[0] = img.copy()

            if combined:
                return img
            else:
                return images

        debugger.width = int(x.shape[2])
        # print('    surrogate: ', surrogate.shape)

        # Calculate new states
        new_states = [None for _ in range(STATE_DROPOUT_BEGIN + 1)]
        is_last_step = (torch.abs(states[:, STATE_STEP_DIM:STATE_STEP_DIM + 1] + 1 - self.cfg.test_steps)
                        < 1e-4).to(torch.float32)
        submitted = is_last_step

        new_states[STATE_REWARD_DIM] = submitted
        new_states[STATE_STOPPED_DIM] = submitted
        # Increment the step
        new_states[STATE_STEP_DIM] = (states[:, STATE_STEP_DIM] + 1)[:, None]

        # Update filter usage
        filter_usage = states[:, STATE_STEP_DIM + 1:]
        # print('usage v.s. onehot', filter_usage.shape, filter_one_hot.shape)
        assert len(filter_usage.shape) == len(filter_one_hot.shape)

        regular_filter_start = 0

        # Penalize submission action that is not the final action.
        early_stop_penalty = (1 - is_last_step) * submitted * self.cfg.early_stop_penalty

        usage_penalty = torch.sum(filter_usage * filter_one_hot[:, regular_filter_start:], dim=1, keepdim=True)
        new_filter_usage = torch.maximum(filter_usage, filter_one_hot[:, regular_filter_start:])
        new_states[STATE_STEP_DIM + 1] = new_filter_usage

        # print("submitted.shape, new_states[STATE_STEP_DIM].shape", submitted.shape, new_states[STATE_STEP_DIM].shape)
        new_states = torch.cat(new_states, dim=1)
        # print('new_states:', new_states.shape)

        if self.cfg.clamp:
            x = torch.clip(x, min=0.0, max=5.0)

        entropy_penalty = (1.0 - progress) * self.cfg.exploration_penalty * (-entropy + math.log(len(self.filters)))

        # Will be substracted from award
        penalty = torch.mean(torch.clip(x - 1, min=0)**2, dim=(1, 2, 3))[:, None] + \
                  entropy_penalty + usage_penalty * self.cfg.filter_usage_penalty + early_stop_penalty

        # print('states, new_states:', states.shape, new_states.shape)
        # print('penalty:', penalty.shape)

        if high_res is None:
            return (x, new_states, surrogate, penalty), debug_info, debugger
        else:
            return (x, new_states, high_res_output), debug_info, debugger

    def forward_de(self, inp, progress, high_res=None):
        train = 1 if self.training else 0
        if self.input_target:
            x, targets, z, states = inp
        else:
            x, z, states = inp

        selection_noise = z[:, 0:1]
        filtered_images = []
        filter_debug_info = []
        high_res_outputs = []

        x_down = self.down_sample(x)
        if self.cfg.shared_feature_extractor:
            if self.input_target:
                targets_down = self.down_sample(targets)
                filter_features = self.feature_extractor(enrich_image_input_w_target(self.cfg, x_down, targets_down, states))
            else:
                filter_features = self.feature_extractor(enrich_image_input(self.cfg, x_down, states))
        else:
            raise ValueError("current just support shared_feature_extractor")
        # filter_features.sum().backward()
        for j, filter in enumerate(self.filters):
            # print('    creating filter:', j, 'name:', str(filter.__class__), 'abbr.', filter.get_short_name())
            # print('      filter_features:', filter_features.shape)
            filtered_image_batch, high_res_output, per_filter_debug_info = filter(x, filter_features, high_res=high_res)
            high_res_outputs.append(high_res_output)
            filtered_images.append(filtered_image_batch)
            filter_debug_info.append(per_filter_debug_info)
            # print('      output:', filtered_image_batch.shape)
            # filtered_image_batch.sum().backward()

        # [batch_size, #filters, H, W, C]
        # for img in filtered_images:
        #     print('img', img.shape)
        filtered_images = torch.stack(filtered_images, dim=1)
        # print('    filtered_images:', filtered_images.shape)

        # filtered_images.sum().backward()
        # action_selection
        if self.input_target:
            selector_features = self.action_selection(enrich_image_input_w_target(self.cfg, x_down, targets_down, states))
        else:
            self.action_selection(enrich_image_input(self.cfg, x_down, states))
        print('    selector features:', selector_features.shape)
        selector_features = self.lrelu(self.fc1(selector_features))

        print('    selector features:', selector_features.shape)
        pdf = self.softmax(self.fc2(selector_features)) + 1e-37
        print('    pdf', pdf.shape)
        print('    pdf_filter', pdf[:, 1:].shape)

        pdf = pdf * (1 - self.cfg.exploration) + self.cfg.exploration * 1.0 / len(self.filters)
        # pdf = tf.to_float(is_train) * tf.concat([pdf[:, :1], pdf[:, 1:] * states[:, STATE_DROPOUT_BEGIN:]], axis=1) \
        # + (1.0 - tf.to_float(is_train)) * pdf
        pdf = pdf / (torch.sum(pdf, dim=1, keepdim=True) + 1e-30)
        '''
        have pdf now
        '''
        entropy = -pdf * torch.log(pdf)
        entropy = torch.sum(entropy, dim=1)[:, None]
        print('    pdf:', pdf.shape)
        print('    entropy:', entropy.shape)
        print('    selection_noise:', selection_noise.shape)
        random_filter_id = pdf_sample(pdf, selection_noise)
        max_filter_id = torch.argmax(pdf, dim=1).to(torch.int32)

        selected_filter_id = (train * random_filter_id + (1 - train) * max_filter_id).to(torch.int64)
        print("selected_filter_id", selected_filter_id, random_filter_id, max_filter_id)
        print('    selected_filter_id:', selected_filter_id.shape)
        '''
        finally we get selected filter
        high res: where im path become useful
        '''

        # selected_filter_id = torch.clip(selected_filter_id, min=0, max=len(self.filters)-1)
        # filter_one_hot = F.one_hot(selected_filter_id, num_classes=len(self.filters))
        filter_one_hot = one_hot(len(self.filters), selected_filter_id)

        print('    filter one_hot', filter_one_hot.shape, filter_one_hot)
        surrogate = torch.sum(filter_one_hot * torch.log(pdf + 1e-10), dim=1, keepdim=True)
        print('    final surrogate', surrogate.shape, surrogate)
        exit()

        '''
        surrogate: to compute gradient of action selection
        '''

        x = torch.sum(filtered_images * filter_one_hot[:, :, None, None, None], dim=1)
        '''
        final x for output -> x will be computed gradient when optimizing para determination
        '''

        if high_res is not None:
            high_res_outputs = torch.stack(high_res_outputs, dim=1)
            high_res_output = torch.sum(high_res_outputs * filter_one_hot[:, :, None, None, None], dim=1)

        # only the first image will get debug_info
        debug_info = {
            'state': states,
            'selected_filter_id': selected_filter_id[0],
            'filter_debug_info': filter_debug_info,
            'pdf': pdf[0],
            'selected_filter': selected_filter_id,
        }

        # Combined: Three in one 64x64 ?
        #           otherwise returns pdf, detail, mask
        def debugger(debug_info, combined=True):
            size = len(self.cfg.filters)  # 8
            img = None
            images = [None for i in range(3)]
            for i, filter in enumerate(self.filters):
                selected = i == debug_info['selected_filter_id']
                if selected:
                    img = filter.visualize_mask(debug_info['filter_debug_info'][i], (64, 64)) * 0.8
            assert img is not None
            if not combined:
                # Mask
                images[2] = img.copy()
                # reset img
                img = img * 0 + 0.5

            c = 0
            for i, filter in enumerate(self.filters):
                pdf = debug_info['pdf'][i]
                if pdf < 1e-10:
                    continue
                else:
                    c += 1
                selected = i == debug_info['selected_filter_id']
                if selected:
                    filter.visualize_filter(debug_info['filter_debug_info'][i], img)
            if not combined:
                # detail
                images[1] = img.copy()
                # reset img
                img = img * 0 + 0.5
            c = 0
            for i, filter in enumerate(self.filters):
                per_col = (len(self.cfg.filters) + 1) // 2  # 4
                x = c // per_col * 30
                y = size * (c % per_col + 1)
                pdf = debug_info['pdf'][i]
                if pdf < 1e-10:
                    continue
                else:
                    c += 1
                cv2.putText(img, filter.get_short_name(), (x + 6, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.233,
                            (255, 255, 255))
                selected = i == debug_info['selected_filter_id']
                color = 1.0 if selected else 0.3
                width = int(pdf * 20)
                height = 0.35
                corners = [(x + 16, int(y + (1 - height) * size // 2)),
                           (x + 16 + width, int(y + (1 + height) * size // 2))]
                cv2.rectangle(img, (corners[0][0] - 1, corners[0][1] - 1),
                              (corners[1][0] + 1, corners[1][1] + 1), (1, 1, 1), cv2.FILLED)
                cv2.rectangle(img, corners[0], corners[1], (color, 0.3, 0.3), cv2.FILLED)
            if not combined:
                # pdf
                images[0] = img.copy()

            if combined:
                return img
            else:
                return images

        debugger.width = int(x.shape[2])
        # print('    surrogate: ', surrogate.shape)

        # Calculate new states
        new_states = [None for _ in range(STATE_DROPOUT_BEGIN + 1)]
        is_last_step = (torch.abs(states[:, STATE_STEP_DIM:STATE_STEP_DIM + 1] + 1 - self.cfg.test_steps)
                        < 1e-4).to(torch.float32)
        # of shape [bs, 1]
        submitted = is_last_step

        new_states[STATE_REWARD_DIM] = submitted
        new_states[STATE_STOPPED_DIM] = submitted
        # Increment the step
        new_states[STATE_STEP_DIM] = (states[:, STATE_STEP_DIM] + 1)[:, None]

        # Update filter usage
        filter_usage = states[:, STATE_STEP_DIM + 1:]
        # print('usage v.s. onehot', filter_usage.shape, filter_one_hot.shape)
        assert len(filter_usage.shape) == len(filter_one_hot.shape)

        regular_filter_start = 0

        # Penalize submission action that is not the final action.
        early_stop_penalty = (1 - is_last_step) * submitted * self.cfg.early_stop_penalty

        usage_penalty = torch.sum(filter_usage * filter_one_hot[:, regular_filter_start:], dim=1, keepdim=True)
        new_filter_usage = torch.maximum(filter_usage, filter_one_hot[:, regular_filter_start:])
        new_states[STATE_STEP_DIM + 1] = new_filter_usage

        # print("submitted.shape, new_states[STATE_STEP_DIM].shape", submitted.shape, new_states[STATE_STEP_DIM].shape)
        new_states = torch.cat(new_states, dim=1)
        # print('new_states:', new_states.shape)

        if self.cfg.clamp:
            x = torch.clip(x, min=0.0, max=5.0)

        entropy_penalty = (1.0 - progress) * self.cfg.exploration_penalty * (-entropy + math.log(len(self.filters)))

        # Will be substracted from award
        penalty = torch.mean(torch.clip(x - 1, min=0)**2, dim=(1, 2, 3))[:, None] + \
                  entropy_penalty + usage_penalty * self.cfg.filter_usage_penalty + early_stop_penalty

        # print('states, new_states:', states.shape, new_states.shape)
        # print('penalty:', penalty.shape)

        if high_res is None:
            return (x, new_states, surrogate, penalty), debug_info, debugger
        else:
            return (x, new_states, high_res_output), debug_info, debugger


if __name__ == "__main__":
    # from easydict import EasyDict
    # cfg = EasyDict({"fc1_size": 4096, "curve_steps": 8})
    # ft = FeatureExtractor((14, 64, 64), 32, 4096, 0.5)
    # x = torch.randn((1, 14, 64, 64))
    # x = ft(x)
    # print(x.shape)
    from config import cfg
    print(cfg.curve_steps)
    batch = 1
    agent = Agent(cfg, (64, 64), 'cpu')
    x = torch.randn((batch, 3, 512, 512))
    z = torch.randn((batch, cfg.z_dim))
    print(z)
    states = torch.randn((batch, cfg.num_state_dim))
    agent((x, z, states), 0.1)
    print(agent.state_dict())
    # torch.save(agent.state_dict(), "agent.pth")





