import glob
import numpy
import scipy.ndimage
from datasets.Util.Reader import create_tensor_dict
from datasets.FeedDataset import OneshotImageDataset
from Log import log

DEFAULT_PATH = "/data/corpora/youtube-objects/youtube_masks/"


def _load_videos(path):
  pattern = path + "*/data/*/*/*/images/"
  folders = sorted(glob.glob(pattern))

  #filter out the 2 sequences, which only have a single annotated frame
  folders = [f for f in folders if "motorbike/data/0007/shots/001" not in f and "cow/data/0019/shots/001" not in f]

  tags = [f.split("/")[-7] + "_" + f.split("/")[-5] for f in folders]

  videos = []
  for folder in folders:
    img_files = sorted(glob.glob(folder + "*.png"))
    label_files = [f.replace("/images/", "/labels/").replace(".png", ".jpg") for f in img_files]
    video = []
    for img_file, label_file in zip(img_files, label_files):
      img_val = scipy.ndimage.imread(img_file) / 255.0
      label_val = scipy.ndimage.imread(label_file) / 255
      label_val = numpy.expand_dims(label_val, 2)
      tag = img_file.split("/")[-7] + "_" + img_file.split("/")[-5] + "/" + img_file.split("/")[-1]
      tensors = create_tensor_dict(unnormalized_img=img_val, label=label_val, tag=tag)
      video.append(tensors)
    videos.append(video)
  return tags, videos


class YoutubeObjectsOneshotDataset(OneshotImageDataset):
  _videos = None

  def _get_video_data(self):
    return self._videos[self._video_idx]

  def __init__(self, config, subset):
    super(YoutubeObjectsOneshotDataset, self).__init__(config, 2, 255, subset, (None, None))
    if YoutubeObjectsOneshotDataset._videos is None:
      print >> log.v1, "loading youtube objects"
      YoutubeObjectsOneshotDataset._videos = _load_videos(DEFAULT_PATH)
      print >> log.v1, "done"
    self._video_tags, self._videos = YoutubeObjectsOneshotDataset._videos
