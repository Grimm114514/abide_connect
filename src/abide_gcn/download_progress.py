"""Terminal progress display for Nilearn's per-file ABIDE downloads."""

from __future__ import annotations

from types import ModuleType

from tqdm import tqdm


class AbideDownloadProgress:
    """Show completed ROI files and aggregate network bytes while Nilearn fetches."""

    def __init__(self, expected_subjects: int = 0) -> None:
        self.expected_subjects = expected_subjects
        self.completed_subjects = 0
        self._fetch_module: ModuleType | None = None
        self._utils_module: ModuleType | None = None
        self._original_fetch_files = None
        self._original_chunk_read = None
        self._subject_bar = None
        self._transfer_bar = None

    def __enter__(self) -> "AbideDownloadProgress":
        from nilearn.datasets import _utils, func

        self._fetch_module = func
        self._utils_module = _utils
        self._original_fetch_files = func.fetch_files
        self._original_chunk_read = _utils._chunk_read_
        self._subject_bar = tqdm(
            total=self.expected_subjects,
            desc="CC200 时序",
            unit="份",
            dynamic_ncols=True,
            position=0,
        )
        self._transfer_bar = tqdm(
            desc="下载流量",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            dynamic_ncols=True,
            position=1,
        )

        def tracked_fetch_files(data_dir, files, *args, **kwargs):
            files = list(files)
            result = self._original_fetch_files(data_dir, files, *args, **kwargs)
            roi_count = sum(
                str(file_path).endswith("_rois_cc200.1D")
                for file_path, _url, _options in files
            )
            if roi_count:
                self.completed_subjects += roi_count
                self._subject_bar.update(roi_count)
            return result

        def tracked_chunk_read(response, local_file, *args, **kwargs):
            original_iter_content = response.iter_content

            def tracked_iter_content(chunk_size):
                for chunk in original_iter_content(chunk_size):
                    if chunk:
                        self._transfer_bar.update(len(chunk))
                    yield chunk

            response.iter_content = tracked_iter_content
            return self._original_chunk_read(
                response, local_file, *args, **kwargs
            )

        func.fetch_files = tracked_fetch_files
        _utils._chunk_read_ = tracked_chunk_read
        return self

    def set_expected_subjects(self, count: int) -> None:
        """Set the exact QC-filtered subject count from Nilearn's phenotype table."""

        self.expected_subjects = count
        self._subject_bar.total = count
        self._subject_bar.refresh()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._fetch_module.fetch_files = self._original_fetch_files
        self._utils_module._chunk_read_ = self._original_chunk_read
        self._subject_bar.close()
        self._transfer_bar.close()
