from .agent import *
from .states import *


class AgentFixPipeline(Agent):
    def __init__(self, cfg, shape=(16, 64, 64), device='cuda', input_target=False, state_inp_func=states_supp_to_inp):
        self.state_inp_func = state_inp_func
        super(AgentFixPipeline, self).__init__(cfg=cfg, shape=shape, device=device, input_target=input_target)

    def forward(self, inp, progress,
                high_res=None,
                specified_filter_id=None,
                inject_noise_dpg=False,
                only_return_agent_out=False):
        train = 1 if self.training else 0
        if self.input_target:
            x, targets, z, states, states_supp = inp
        else:
            x, z, states, states_supp = inp

        selection_noise = z[:, 0:1]
        filtered_images = []
        filter_debug_info = []
        high_res_outputs = []
        batch_size = x.shape[0]

        x_down = self.down_sample(x)
        if self.cfg.shared_feature_extractor:
            if self.input_target:
                targets_down = self.down_sample(targets)
                filter_features = self.feature_extractor(
                    enrich_image_input_w_target(
                        self.cfg,
                        x_down,
                        targets_down,
                        states=self.state_inp_func(states_supp)
                    )
                )
            else:
                filter_features = self.feature_extractor(
                    enrich_image_input(
                        self.cfg, x_down,
                        states=self.state_inp_func(states_supp)
                    )
                )
        else:
            raise ValueError("current just support shared_feature_extractor")

        filter_param = None
        agent_out_batch = None
        filter_params_batch = [state_supp.params for state_supp in states_supp]
        # filter_params_batch -> list of size (batch_size, filter_num)
        for j, filter in enumerate(self.filters):
            # print('    creating filter:', j, 'name:', str(filter.__class__), 'abbr.', filter.get_short_name())
            # print('      filter_features:', filter_features.shape)
            if self.cfg.isp_inp_last_step:
                isp_inp = x
            else:
                # input original input image
                isp_inp = states_supp_to_isp_inp(states_supp)
            steps_batch = [state_supp.step for state_supp in states_supp]
            filtered_image_batch, high_res_output, per_filter_debug_info = filter(
                isp_inp, filter_features,
                high_res=high_res,
                inject_noise=inject_noise_dpg,
                steps_batch=steps_batch
            )
            high_res_outputs.append(high_res_output)
            filtered_images.append(filtered_image_batch)
            filter_debug_info.append(per_filter_debug_info)
            if specified_filter_id is not None and j == specified_filter_id:
                filter_param = per_filter_debug_info['filter_parameters_batch']
                agent_out_batch = filter_param
                if only_return_agent_out:
                    return agent_out_batch
                # print("update filter param batch")
                # print("filter_param", filter_param)
                # print("filter_params_batch", filter_params_batch)
                for idx_batch in range(batch_size):
                    temp_device = filter_params_batch[idx_batch][j].device
                    filter_params_batch[idx_batch][j] = filter_param[idx_batch].to(temp_device)
                # print("filter_params_batch after", filter_params_batch)

            # print('      output:', filtered_image_batch.shape)
            # filtered_image_batch.sum().backward()

        # [batch_size, #filters, H, W, C]
        # for img in filtered_images:
        #     print('img', img.shape)
        filtered_images = torch.stack(filtered_images, dim=1)
        # print('    filtered_images:', filtered_images.shape)

        selected_filter_id = torch.zeros((batch_size)).to(x.device)
        if specified_filter_id is None:
            if self.input_target:
                selector_features = self.action_selection(enrich_image_input_w_target(self.cfg, x_down, targets_down, states))
            else:
                selector_features = self.action_selection(enrich_image_input(self.cfg, x_down, states))
            selector_features = self.lrelu(self.fc1(selector_features))

            pdf = self.softmax(self.fc2(selector_features)) + 1e-37

            pdf = pdf * (1 - self.cfg.exploration) + self.cfg.exploration * 1.0 / len(self.filters)
            pdf = pdf / (torch.sum(pdf, dim=1, keepdim=True) + 1e-30)
            entropy = -pdf * torch.log(pdf)
            entropy = torch.sum(entropy, dim=1)[:, None]
            random_filter_id = pdf_sample(pdf, selection_noise)
            max_filter_id = torch.argmax(pdf, dim=1).to(torch.int32)

            selected_filter_id = (train * random_filter_id + (1 - train) * max_filter_id).to(torch.int64)
        else:
            # todo fake pdf entropy
            pdf = torch.randn(x.shape[0], len(self.filters)).to(x.device)
            pdf = pdf * (1 - self.cfg.exploration) + self.cfg.exploration * 1.0 / len(self.filters)
            pdf = pdf / (torch.sum(pdf, dim=1, keepdim=True) + 1e-30)
            entropy = -pdf * torch.log(pdf)
            entropy = torch.sum(entropy, dim=1)[:, None]

        if specified_filter_id is not None:
            assert specified_filter_id in range(0, len(self.filters))
            selected_filter_id[:] = specified_filter_id
            selected_filter_id = selected_filter_id.to(torch.int64)
            # print(selected_filter_id)

        filter_one_hot = one_hot(len(self.filters), selected_filter_id)

        # print('    filter one_hot', filter_one_hot.shape, filter_one_hot)
        # surrogate = torch.sum(filter_one_hot * torch.log(pdf + 1e-10), dim=1, keepdim=True)
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
            'action_batch': agent_out_batch
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

        '''
        save new state supp now
        '''
        new_states_supp = get_updated_states_supp(
            states_supp,
            params_batch=filter_params_batch,
            stopped=submitted
        )

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
        penalty = (torch.mean(torch.clip(x - 1, min=0)**2, dim=(1, 2, 3))[:, None] +
                   entropy_penalty +
                   usage_penalty * self.cfg.filter_usage_penalty +
                   early_stop_penalty)
        # TODO!!!! penalty term from SPG is actually degrading DPG
        # TODO!!!! figure this out
        # TODO!!!! -> to alleviate this, check out DPG's panelty term optimized for param sel
        penalty = (torch.mean(torch.clip(x - 1, min=0) ** 2, dim=(1, 2, 3))[:, None] +
                   early_stop_penalty)
        # print('states, new_states:', states.shape, new_states.shape)
        # print('penalty:', penalty.shape)

        if high_res is None:
            return (x, new_states, new_states_supp, surrogate, penalty), debug_info, debugger
        else:
            return (x, new_states, new_states_supp, high_res_output), debug_info, debugger

