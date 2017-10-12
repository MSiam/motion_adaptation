from datasets.Dataset import ImageDataset
from datasets.Util.Util import username
from datasets.Util import Reader

VOID_LABEL = 255
PASCAL_VOC_DEFAULT_PATH = "/fastwork/" + username() + "/mywork/data/PascalVOC/benchmark_RELEASE/dataset/"
INPUT_SIZE = (None, None)


class PascalVOCDataset(ImageDataset):
  def __init__(self, config, subset, coord, label_postproc_fn, name, num_classes,
               label_load_fn=Reader.load_label_default, img_load_fn=Reader.load_img_default,
               fraction=1.0, ignore_classes=[]):
    super(PascalVOCDataset, self).__init__(name, PASCAL_VOC_DEFAULT_PATH, num_classes,
                                           config, subset, coord, INPUT_SIZE, VOID_LABEL, fraction,
                                           label_postproc_fn=label_postproc_fn,
                                           label_load_fn=label_load_fn,
                                           img_load_fn=img_load_fn,
                                           ignore_classes=ignore_classes)

  def read_inputfile_lists(self):
    data_list = "train.txt" if self.subset == "train" else "val.txt"
    data_list = "datasets/PascalVOC/" + data_list
    imgs = []
    ans = []
    with open(data_list) as f:
      for l in f:
        im, an = l.strip().split()
        im = self.data_dir + im
        an = self.data_dir + an
        imgs.append(im)
        ans.append(an)
    return imgs, ans
