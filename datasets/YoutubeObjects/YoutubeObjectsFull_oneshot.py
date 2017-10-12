import glob
import numpy
import scipy.ndimage
from scipy.misc import imresize
from datasets.Util.Reader import create_tensor_dict
from datasets.FeedDataset import OneshotImageDataset
from Log import log

DEFAULT_PATH = "/data/corpora/youtube-objects/youtube_masks_full/"


def _load_video(folder):
  label_files = sorted(glob.glob(folder + "*.jpg"))
  assert len(label_files) > 0

  indices = [int(f.split("/")[-1][:-4]) for f in label_files]
  min_idx = indices[0]
  assert min_idx == min(indices)
  max_idx = max(indices)

  img_folder = folder.replace("/labels/", "/images/")
  video = []
  shape = None

  for idx in xrange(min_idx, max_idx + 1):
    img_file = img_folder + ("frame%04d.jpg" % idx)
    if idx in indices:
      label_file = label_files[indices.index(idx)]
      label_val = scipy.ndimage.imread(label_file) / 255
      label_val = numpy.expand_dims(label_val, 2)
      shape = label_val.shape
    else:
      assert shape is not None
      label_val = numpy.zeros(shape, dtype=numpy.uint8)

    img_val = scipy.ndimage.imread(img_file) #/ 255.0
    #scale down to match label here!
    assert shape is not None
    #assert img_val.shape[0] == 2 * shape[0]
    #assert img_val.shape[1] == 2 * shape[1]
    img_val = imresize(img_val, shape[:2])
    img_val = img_val.astype("float32") / 255.0

    tag = img_file.split("/")[-7] + "_" + img_file.split("/")[-5] + "/" + img_file.split("/")[-1]
    if idx not in indices:
      tag += "_DO_NOT_STORE_LOGITS"
    tensors = create_tensor_dict(unnormalized_img=img_val, label=label_val, tag=tag)
    video.append(tensors)
  return video


class YoutubeObjectsFullOneshotDataset(OneshotImageDataset):
  cache = None

  def _get_video_data(self):
    return self._video

  def set_video_idx(self, video_idx):
    if self._video is None or self.get_video_idx() != video_idx:
      if YoutubeObjectsFullOneshotDataset.cache is not None and YoutubeObjectsFullOneshotDataset.cache[0] == video_idx:
        self._video = YoutubeObjectsFullOneshotDataset.cache[1]
      else:
        #load video
        print >> log.v2, "loading sequence", video_idx, self.video_tag(video_idx), "of youtubeobjects (full) ..."
        self._video = _load_video(self._folders[video_idx])
        print >> log.v2, "done"
        YoutubeObjectsFullOneshotDataset.cache = (video_idx, self._video)
      self._video_idx = video_idx

  def __init__(self, config, subset):
    super(YoutubeObjectsFullOneshotDataset, self).__init__(config, 2, 255, subset, (None, None))
    self._video = None

    path = DEFAULT_PATH
    pattern = path + "*/data/*/*/*/labels/"
    folders = sorted(glob.glob(pattern))

    # filter out the 2 sequences, which only have a single annotated frame
    self._folders = [f for f in folders if "motorbike/data/0007/shots/001" not in f and "cow/data/0019/shots/001" not in f]
    self._video_tags = [f.split("/")[-7] + "_" + f.split("/")[-5] for f in self._folders]

    self.set_video_idx(0)
