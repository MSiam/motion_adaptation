import glob
import scipy.ndimage
from datasets.Util.Reader import create_tensor_dict
from datasets.FeedDataset import OneshotImageDataset
from Log import log

DEFAULT_PATH = "/data/corpora/SegTrackv2/"
SEQUENCES = ["bird_of_paradise", "birdfall", "girl", "monkeydog/1", "monkeydog/2", "parachute", "penguin/1",
             "penguin/2", "penguin/3", "penguin/4", "penguin/5", "penguin/6", "cheetah/1", "cheetah/2", "worm",
             "bmx/1", "bmx/2", "drift/1", "drift/2", "frog", "hummingbird/1", "hummingbird/2", "monkey", "soldier"]
BMP_SEQUENCES = ["cheetah", "girl", "monkeydog", "penguin"]
GT_BMP_SEQUENCES = ["girl", "cheetah/1"]


def _load_videos(path):
  tags = SEQUENCES
  folders = [path + "GroundTruth/" + s + "/" for s in SEQUENCES]
  img_sets = [path + "ImageSets/" + s.split("/")[0] + ".txt" for s in SEQUENCES]

  videos = []
  for seq, folder, img_set in zip(SEQUENCES, folders, img_sets):
    if seq in GT_BMP_SEQUENCES:
      ending = ".bmp"
    else:
      ending = ".png"

    fns = [l.strip() for l in open(img_set).readlines()[1:]]
    #TODO

    label_files = [folder + fn + ending for fn in fns]
    assert len(label_files) > 0
    img_files = [f.replace("/GroundTruth/", "/JPEGImages/").replace(seq, seq.split("/")[0]) for f in label_files]
    if seq.split("/")[0] in BMP_SEQUENCES:
      img_files = [f.replace(".png", ".bmp") for f in img_files]
    video = []
    for img_file, label_file in zip(img_files, label_files):
      img_val = scipy.ndimage.imread(img_file) / 255.0
      label_val = scipy.ndimage.imread(label_file) / 255
      label_val = label_val[..., :1]
      assert label_val.ndim == 3, label_val.ndim
      #label_val = numpy.expand_dims(label_val, 2)
      tag = seq.replace("/", "_") + "/" + label_file.split("/")[-1]
      tensors = create_tensor_dict(unnormalized_img=img_val, label=label_val, tag=tag)
      video.append(tensors)
    videos.append(video)
  return tags, videos


class Segtrackv2OneshotDataset(OneshotImageDataset):
  _videos = None

  def _get_video_data(self):
    return self._videos[self._video_idx]

  def __init__(self, config, subset):
    super(Segtrackv2OneshotDataset, self).__init__(config, 2, 255, subset, (None, None))
    if Segtrackv2OneshotDataset._videos is None:
      print >> log.v1, "loading SegTrackv2"
      Segtrackv2OneshotDataset._videos = _load_videos(DEFAULT_PATH)
      print >> log.v1, "done"
    self._video_tags, self._videos = Segtrackv2OneshotDataset._videos
