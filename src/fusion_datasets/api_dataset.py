from typing import Any

import requests
from kedro_datasets.api.api_dataset import APIDataset as OriginalAPIDataset


class APIDataset(OriginalAPIDataset):  # noqa: D101
    def _execute_save_with_chunks(
        self,
        json_data: list[dict[str, Any]],
    ) -> requests.Response:
        chunk_size = self._chunk_size
        unwrap = False
        if chunk_size == -1:
            chunk_size = 1
            unwrap = True
        # NOTE has fixed the issue with the original code when chunk_size is 1
        n_chunks = (len(json_data) + chunk_size - 1) // chunk_size

        response = None
        if unwrap:
            for i in range(len(json_data)):
                response = self._execute_save_request(json_data=json_data[i])
        else:
            for i in range(n_chunks):
                send_data = json_data[i * chunk_size : (i + 1) * chunk_size]
                response = self._execute_save_request(json_data=send_data)
        return response
