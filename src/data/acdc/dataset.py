from typing import Callable, List, Tuple

import h5py
from torchvision.datasets import VisionDataset
from torchvision.transforms import ToTensor, transforms

from src.data.transforms import NormalizeSample, SegmentationToTensor


class Acdc(VisionDataset):
    """Implementation of torchvision's ``VisionDataset`` for the ACDC dataset.

    Args:
        path: Path to the HDF5 dataset.
        image_set: One of "train", "val" or "test", indicating the subset of images to use.
        transform: a function/transform that takes in a numpy array and returns a transformed version.
        target_transform: a function/transform that takes in the target and transforms it.
    """

    def __init__(
        self,
        path: str,
        image_set: str,
        transform: Callable = None,
        target_transform: Callable = None,
    ):
        transform = (
            transforms.Compose([ToTensor(), NormalizeSample()])
            if not transform
            else transforms.Compose([transform, ToTensor()])
        )
        target_transform = (
            transforms.Compose([SegmentationToTensor()])
            if not target_transform
            else transforms.Compose([target_transform, SegmentationToTensor()])
        )

        super().__init__(path, transform=transform, target_transform=target_transform)

        self.image_set = image_set
        self.item_list = self._get_instant_paths()

    def __len__(self):  # noqa: D105
        return len(self.item_list)

    def list_groups(self, level: str = "instant") -> List[str]:
        """Lists the paths of the different levels of groups/clusters data samples in ``self.image_set`` can belong to.

        Args:
            level: One of "patient" or "instant", indicating the hierarchical level at which to group data samples.
                - "patient": all the data from the same patient is associated to a unique ID.
                - "instant": all the data from the same instant of a patient is associated to a unique ID.

        Returns:
            IDs of the different levels of groups/clusters data samples in ``self.image_set`` can belong to.
        """
        with h5py.File(self.root, "r") as dataset:
            # List the patients
            groups = [f"{self.image_set}/{patient_id}" for patient_id in dataset[self.image_set].keys()]

            if level == "instant":
                groups = [f"{patient}/{instant}" for patient in groups for instant in dataset[patient].keys()]

        return groups

    def _get_instant_paths(self) -> List[Tuple[str, int]]:
        """Lists paths to the instants, from the requested ``image_set``, inside the HDF5 file.

        Returns:
            paths to the instants, from the requested ``image_set``, inside the HDF5 file.
        """
        image_paths = []
        instant_paths = self.list_groups(level="instant")
        with h5py.File(self.root, "r") as dataset:
            for instant_path in instant_paths:
                num_slices = len(dataset[instant_path]["img"])
                image_paths.extend((instant_path, slice) for slice in range(num_slices))

        return image_paths

    def __getitem__(self, index):
        """Fetches data required for a single image/groundtruth pair.

        Args:
            index: index of the sample in the subset's ``item_list``.

        Returns:
            data for training on a single image/groundtruth item.
        """
        set_patient_instant_key, slice = self.item_list[index]

        with h5py.File(self.root, "r") as dataset:
            # Collect and process data
            img = dataset[set_patient_instant_key]["img"][slice][()]
            gt = dataset[set_patient_instant_key]["gt"][slice][()]

        img = self.transform(img)
        gt = self.target_transform(gt).squeeze()

        return img, gt
