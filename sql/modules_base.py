import torch
import texar.torch as tx
from typing import Tuple, Dict, Union, Any, Optional, Type, Callable

from sql.misc_utils import unionize_dicts
from sql import utils as sql_utils
from sql import losses as sql_losses
from sql.utils import ForwardMode
from sql.rewards import reward_name_to_cls_map
from sql.types import BatchType, HF_BatchType, FloatTensor, LongTensor


class SoftQModelBase(torch.nn.Module):
    """
    When the model is a Soft Q-Learning model,
        model:       Q-model
        model_:      Target Q-model
        actor_model: None

    When the model is a Soft Actor-Critic model,
        model:       Critic Q-model
        model_:      Critic Target Q-model
        actor_model: Actor pi-model

    Otherwise,
        model:       model (MLE, Policy Gradient, etc)
        model_:      None
        actor_model: None
    """

    def __init__(
            self,
            model_constructor: Callable[[], torch.nn.Module],
            target_model_constructor: Optional[Callable[[], torch.nn.Module]],
            actor_model_constructor: Optional[Callable[[], Optional[torch.nn.Module]]],
            behavior_model_constructor: Optional[Callable[[], Optional[torch.nn.Module]]],
            sql_loss_impl: str,
            target_update_method: Optional[str],
            target_learning_rate: float,
            reward_shaping: bool,
            reward_shaping_old_min: float,
            reward_shaping_old_max: float,
            reward_shaping_new_min: float,
            reward_shaping_new_max: float,
            sql_loss_coefficients: Optional[float] = None,
            sql_loss_margin_constant: Optional[float] = None,
            sql_loss_margin_coefficient: Optional[float] = None,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
            beam_width: Optional[int] = None,
            reward_name: Optional[str] = None,
            # TST arguments
            tst_dataset: Optional[str] = None,
            tst_data_seed: Optional[int] = None,
            # Classification arguments
            LM_type: str = 'gpt2',
            experiment: str = 'Test',
            experiment_seed: int = 0,
            kshot: int = -1,
            task_name: str = 'SST-2',
    ) -> None:
        super().__init__()
        if (target_update_method is not None and
                target_update_method not in ["copy", "polyak"]):
            raise ValueError(f"`target_update_method`={target_update_method}")

        if top_k is not None and top_p is not None:
            raise ValueError

        if target_model_constructor is None:
            target_model_constructor = model_constructor

        if actor_model_constructor is None:
            actor_model_constructor = lambda: None

        if behavior_model_constructor is None:
            behavior_model_constructor = lambda: None

        # Construct the model and the target model
        self._model = model_constructor()
        self._model_ = target_model_constructor()
        self._actor_model = actor_model_constructor()
        self._behavior_model = behavior_model_constructor()

        # Initialize the target model with the
        # model's current parameters if applicable.
        if target_update_method in ["copy", "polyak"]:
            self._model_.load_state_dict(
                self._model.state_dict())

        self._sql_loss_impl = sql_loss_impl
        self._target_update_method = target_update_method
        self._target_learning_rate = target_learning_rate
        self._sql_loss_coefficients = sql_loss_coefficients
        self._sql_loss_margin_constant = sql_loss_margin_constant
        self._sql_loss_margin_coefficient = sql_loss_margin_coefficient
        self._top_k = top_k
        self._top_p = top_p
        self._beam_width = beam_width

        if reward_name is not None:
            if reward_name in ['gpt2-sentiment-bleu-no-input', 'plm-classifier']:
                self._reward_function = \
                    reward_name_to_cls_map[reward_name](dataset=tst_dataset,
                                                        dataset_seed=tst_data_seed,
                                                        # LM_type=LM_type,
                                                        # experiment=experiment,
                                                        # experiment_seed=experiment_seed,
                                                        kshot=kshot,
                                                        # task_name=task_name
                                                        )
            else:
                self._reward_function = reward_name_to_cls_map[reward_name]()
            
        else:
            self._reward_function = None

        if reward_shaping is True:
            self._reward_shaping_func = sql_utils.get_reward_shaping_func(
                old_min=reward_shaping_old_min,
                old_max=reward_shaping_old_max,
                new_min=reward_shaping_new_min,
                new_max=reward_shaping_new_max)
        else:
            self._reward_shaping_func = lambda _r: _r

    @property
    def is_actor_critic(self) -> bool:
        return self._actor_model is not None

    def sync_target_model(self) -> None:
        # Do nothing
        if self._target_update_method is None:
            return

        # https://github.com/transedward/pytorch-dqn/blob/master/dqn_learn.py#L221
        if self._target_update_method == "copy":
            self._model_.load_state_dict(
                self._model.state_dict())

        # Target network update
        # Note that we are assuming `model.parameters()`
        # would yield the same parameter orders.
        # https://towardsdatascience.com/double-deep-q-networks-905dd8325412
        if self._target_update_method == "polyak":
            for param_, param in zip(
                    self._model_.parameters(),
                    self._model.parameters()):

                param_.data.copy_(
                    (1 - self._target_learning_rate) * param_ +
                    self._target_learning_rate * param)

    def _decode_teacher_forcing(
            self,
            batch: Union[BatchType, HF_BatchType],
            use_target: bool,
    ) -> Tuple[FloatTensor,
               Optional[FloatTensor],
               Optional[FloatTensor],
               LongTensor,
               Optional[LongTensor],
               LongTensor,
               LongTensor]:
        """
        Returns:
            logits:             [batch_size, sequence_length, vocab_size]
            logits_:            [batch_size, sequence_length, vocab_size]
            logits_pi:          [batch_size, sequence_length, vocab_size]
            outputs_id:         [batch_size, sequence_length]
            sampled_outputs_id: [batch_size, sequence_length]
            labels:             [batch_size, sequence_length]
            sequence_lengths:   [batch_size]
        """
        raise NotImplementedError

    def _decode_behavior_forcing(
            self,
            batch: Union[BatchType, HF_BatchType],
            use_target: bool,
    ) -> Tuple[FloatTensor,
               Optional[FloatTensor],
               Optional[FloatTensor],
               LongTensor,
               Optional[LongTensor],
               LongTensor,
               LongTensor]:
        """
        Returns:
            logits:             [batch_size, sequence_length, vocab_size]
            logits_:            [batch_size, sequence_length, vocab_size]
            logits_pi:          [batch_size, sequence_length, vocab_size]
            outputs_id:         [batch_size, sequence_length]
            sampled_outputs_id: [batch_size, sequence_length]
            labels:             [batch_size, sequence_length]
            sequence_lengths:   [batch_size]
        """
        raise NotImplementedError

    def _decode_sampling(
            self,
            batch: Union[BatchType, HF_BatchType],
            use_target: bool,
    ) -> Tuple[FloatTensor,
               Optional[FloatTensor],
               Optional[FloatTensor],
               LongTensor,
               Optional[LongTensor],
               LongTensor,
               LongTensor]:
        """
        Returns:
            logits:             [batch_size, sequence_length, vocab_size]
            logits_:            [batch_size, sequence_length, vocab_size]
            logits_pi:          [batch_size, sequence_length, vocab_size]
            outputs_id:         [batch_size, sequence_length]
            sampled_outputs_id: [batch_size, sequence_length]
            labels:             [batch_size, sequence_length]
            sequence_lengths:   [batch_size]
        """
        raise NotImplementedError

    def _compute_rewards(
            self,
            batch: Union[BatchType, HF_BatchType],
            outputs_id: LongTensor,
            labels: LongTensor,
            sequence_lengths: LongTensor,
    ) -> Tuple[FloatTensor, FloatTensor, Dict[str, Any]]:
        """
        Returns:
            raw_rewards: [batch_size]
            shaped_rewards: [batch_size]

        """
        raise NotImplementedError

    def _forward_MLE(
            self,
            batch: Union[BatchType, HF_BatchType],
    ) -> Tuple[FloatTensor, FloatTensor, Dict]:

        (logits,
         _,
         _,
         _,
         _,
         labels,
         sequence_lengths) = self._decode_teacher_forcing(
            batch=batch,
            use_target=False)

        mle_loss = tx.losses.sequence_sparse_softmax_cross_entropy(
            labels=labels,
            logits=logits,
            sequence_length=sequence_lengths)

        entropy = tx.losses.sequence_entropy_with_logits(
            logits=logits,
            sequence_length=sequence_lengths,
            average_across_timesteps=True,
            sum_over_timesteps=False)

        return logits, mle_loss, {
            "mle/loss": mle_loss,
            "mle/entropy/mean": entropy,
        }

    def _forward_decoding(
            self,
            batch: Union[BatchType, HF_BatchType],
    ) -> Any:
        raise NotImplementedError

    def _forward_PG(
            self,
            batch: Union[BatchType, HF_BatchType],
    ) -> Tuple[FloatTensor, FloatTensor, Dict]:

        (logits,
         _,
         _,
         outputs_id,
         _,
         labels,
         sequence_lengths) = self._decode_sampling(
            batch=batch,
            use_target=False)

        raw_rewards, shaped_rewards, rewards_log = self._compute_rewards(
            batch=batch,
            outputs_id=outputs_id,
            labels=labels,
            sequence_lengths=sequence_lengths)

        pg_loss = tx.losses.pg_loss_with_logits(
            actions=outputs_id,
            logits=logits,
            # [batch_size, 1]
            advantages=shaped_rewards.view(-1, 1),
            batched=True,
            sequence_length=sequence_lengths)

        entropy = tx.losses.sequence_entropy_with_logits(
            logits=logits,
            sequence_length=sequence_lengths,
            average_across_timesteps=True,
            sum_over_timesteps=False)

        return logits, pg_loss, {
            "pg/loss": pg_loss,
            "pg/rewards/raw": raw_rewards.mean(),
            "pg/rewards/shaped": shaped_rewards.mean(),
            "pg/entropy/mean": entropy,
        }

    def _forward_SQL(
            self,
            mode: ForwardMode,
            batch: Union[BatchType, HF_BatchType],
    ) -> Tuple[FloatTensor, FloatTensor, Dict]:

        if mode == ForwardMode.SQL_OFF_GT:
            (logits,
             logits_,
             logits_pi,
             outputs_id,
             sampled_outputs_id,
             labels,
             sequence_lengths) = self._decode_teacher_forcing(
                batch=batch,
                use_target=True)

        if mode == ForwardMode.SQL_OFF_BEHAVIOR:
            (logits,
             logits_,
             logits_pi,
             outputs_id,
             sampled_outputs_id,
             labels,
             sequence_lengths) = self._decode_behavior_forcing(
                batch=batch,
                use_target=True)

        if mode == ForwardMode.SQL_ON:
            (logits,
             logits_,
             logits_pi,
             outputs_id,
             sampled_outputs_id,
             labels,
             sequence_lengths) = self._decode_sampling(
                batch=batch,
                use_target=True)

        raw_rewards, shaped_rewards, rewards_log = self._compute_rewards(
            batch=batch,
            outputs_id=outputs_id,
            labels=labels,
            sequence_lengths=sequence_lengths)

        sql_loss, sql_loss_log = sql_losses.soft_q_loss_with_sparse_rewards(
            implementation=self._sql_loss_impl,
            logits=logits,
            logits_=logits_,
            logits_pi=logits_pi,
            actions=outputs_id,
            sampled_actions=sampled_outputs_id,
            rewards=shaped_rewards,
            sequence_length=sequence_lengths,
            coefficient=self._sql_loss_coefficients,
            # Do not add margin losses unless the
            # actions are ground truth actions.
            margin_constant=(
                self._sql_loss_margin_constant
                if mode == ForwardMode.SQL_OFF_GT else None),
            margin_coefficient=(
                self._sql_loss_margin_coefficient
                if mode == ForwardMode.SQL_OFF_GT else None))

        sql_utils.add_prefix_to_dict_keys_inplace(
            rewards_log, prefix=f"{mode.value}/rewards/")
        sql_utils.add_prefix_to_dict_keys_inplace(
            sql_loss_log, prefix=f"{mode.value}/")

        sql_loss_log = unionize_dicts([
            rewards_log,
            sql_loss_log,
            {
                f"{mode.value}/rewards/raw": raw_rewards.mean(),
                f"{mode.value}/rewards/shaped": shaped_rewards.mean(),
            },
        ])

        return logits, sql_loss, sql_loss_log

    def _forward(
            self,
            mode: ForwardMode,
            batch: Union[BatchType, HF_BatchType],
    ) -> Tuple[FloatTensor, FloatTensor, Dict]:

        if mode == ForwardMode.MLE:
            return self._forward_MLE(batch=batch)

        # if mode == ForwardMode.INFER:
        #     return self._forward_decoding(batch=batch)

        if mode == ForwardMode.PG:
            return self._forward_PG(batch=batch)

        if mode in [ForwardMode.SQL_ON,
                    ForwardMode.SQL_OFF_GT,
                    ForwardMode.SQL_OFF_RB,
                    ForwardMode.SQL_OFF_BEHAVIOR]:
            return self._forward_SQL(mode=mode, batch=batch)

        raise ValueError(f"Unknown mode {mode}")
