from DAVIS.DAVIS import DAVISDataset, DAVIS2017Dataset
from datasets.DAVIS.DAVIS_oneshot import DavisOneshotDataset
from datasets.DAVIS.DAVIS_contlabels import DavisContDataset

def load_dataset(config, subset, session, coordinator):
  name = config.unicode("dataset").lower()
  task = config.unicode("task", "")
  if task in ("baseline", "teach"):
    if name == "davis":
      return DavisOneshotDataset(config, subset, use_old_label=False)
    else:
      assert False, "Unknown dataset for oneshot: " + name
  elif task == "teachcont":
      return DavisContDataset(config, subset, use_old_label=False)

  if name == "davis":
    return DAVISDataset(config, subset, coordinator)
  else:
    assert False, "Unknown dataset " + name
