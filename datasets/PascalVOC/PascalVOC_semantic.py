from datasets.PascalVOC.PascalVOC import PascalVOCDataset


NUM_CLASSES = 21


class PascalVOCSemanticDataset(PascalVOCDataset):
  def __init__(self, config, subset, coord, fraction=1.0):
    super(PascalVOCSemanticDataset, self).__init__(config, subset, coord,
                                                   label_postproc_fn=lambda x: x,
                                                   name="pascalvoc_semantic",
                                                   num_classes=NUM_CLASSES, fraction=fraction,
                                                   ignore_classes=[0])
