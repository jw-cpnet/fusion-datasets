import datetime
from logging import getLogger
from typing import Callable, Optional

from dateutil.relativedelta import relativedelta

logger = getLogger(__name__)


def init_checkpoint(func):
    def wrapper(self, *args, **kwargs):
        partition_id, checkpoint = args
        if isinstance(checkpoint, str):
            checkpoint = checkpoint.strip()
        new_args = (partition_id, checkpoint)
        self._init_checkpoint(*new_args)
        return func(self, *new_args, **kwargs)

    return wrapper


class ComparisonFunction:
    def __init__(
        self,
        parser: Callable[[str], datetime.datetime],
    ):
        self._current_checkpoint = None
        self.parser = parser

    def _init_checkpoint(self, partition_id: str, checkpoint: str):
        # Set the current checkpoint only if it hasn't been set before
        if self._current_checkpoint is None:
            # Use the partition_id as the checkpoint if the provided checkpoint is empty or None
            self._current_checkpoint = partition_id if not checkpoint else checkpoint

    @init_checkpoint
    def same_month(
        self,
        partition_id: str,
        checkpoint: str,
        increment_method: Optional[str] = "auto",
    ) -> bool:
        """
        A comparison function to enable loading of partitions within the same month. That is, if ``partition_id`` is
        in the same month as ``checkpoint``, return True. Otherwise, return False.

        If the checkpoint is an empty string (you can set the checkpoint to empty string by providing an empty
        CHECKPOINT file, or just use the force_checkpoint key in the data catalog), the comparison will start from
        the first partition with respect to the above rule.

        if there is no CHECKPOINT file, this function will not be called at all. And the pipeline will try to read as
        many partitions as possible. You can also achieve the same result by setting force_checkpoint key to null.

        Args:
            partition_id: partition to compare
            checkpoint: current checkpoint
            increment_method: "auto" (default), "next" or None ("no", "null", "none")
              If set to "auto", will allow partitions from the same month and next month; If set to "next",
              will only use partitions from the next month; If set to None, only partitions from the same
              month as the checkpoint will be used.

        Returns:
            True or False

        """
        load = False
        increment_method = increment_method or ""
        if not increment_method.lower() in ["no", "null", "none", "n", "auto", "next"]:
            raise ValueError("Invalid increment_method!")
        try:
            # month of current checkpoint
            chk_pt = self.parser(self._current_checkpoint)
            # month of this partition id
            part_id = self.parser(partition_id)
            is_same_month = part_id == chk_pt
            is_next_month = part_id == chk_pt + relativedelta(months=1)
            if (
                increment_method is None
                or isinstance(increment_method, str)
                and increment_method.lower() in ("no", "null", "none")
            ):
                # only allow partitions from the same month as chk_pt
                load = is_same_month
            elif increment_method == "auto":
                # same month or next month
                load = is_same_month or is_next_month
            elif increment_method == "next":
                # only allow partitions from next month (of chk_pt)
                load = is_next_month

        except Exception as e:
            raise ValueError("Parser error. Please check your parser")
        # TODO use decorator logger here
        logger.info(
            f"checkpoint: {self._current_checkpoint} "
            f"| partition: {partition_id} | load: {load}"
        )
        return load

    @init_checkpoint
    def boundary(
        self,
        partition_id: str,
        checkpoint: str,
        left: int = None,
        right: int = None,
        close: str = "both",
    ):
        if close not in ["left", "right", "both", None]:
            raise ValueError(f"Invalid argument. `close` got invalid value {close}")
        close = close or "both"
        chk_pt = self.parser(self._current_checkpoint)
        part_id = self.parser(partition_id)

        lower_bound = chk_pt - datetime.timedelta(days=left) if left else chk_pt
        upper_bound = chk_pt + datetime.timedelta(days=right) if right else None
        left_comp = (
            part_id >= lower_bound if close != "right" else part_id > lower_bound
        )
        if upper_bound:
            right_comp = (
                part_id <= upper_bound if close != "left" else part_id < upper_bound
            )
        else:
            right_comp = True

        load = left_comp and right_comp
        logger.debug(
            f"checkpoint: {self._current_checkpoint} "
            f"| partition: {partition_id} | load: {load}"
        )
        return load
