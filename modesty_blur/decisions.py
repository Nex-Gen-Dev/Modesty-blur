"""Per-track blur policy. Pure Python, no ML imports, easy to unit test.

Policy is FAIL-CLOSED:
  * a track is blurred from its very first frame (until proven otherwise);
  * it is only un-blurred if it is a confirmed family member, or it has been
    classified "not female" confidently, repeatedly;
  * anything uncertain stays blurred.
"""

from dataclasses import dataclass


@dataclass
class TrackState:
    ema_female: float = 1.0        # smoothed P(female); starts at "blur me"
    n_obs: int = 0                 # number of classifications so far
    cleared_male: bool = False     # confidently classified as not female
    family_hits: int = 0           # consecutive family face matches
    is_family: bool = False
    last_cls_frame: int = -10**9
    last_fam_frame: int = -10**9
    family_name: str = ""


class DecisionEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tracks: dict[int, TrackState] = {}

    def state(self, tid: int) -> TrackState:
        if tid not in self.tracks:
            self.tracks[tid] = TrackState()
        return self.tracks[tid]

    # ---- scheduling ------------------------------------------------------
    def needs_classification(self, tid: int, frame_idx: int) -> bool:
        s = self.state(tid)
        return s.n_obs == 0 or frame_idx - s.last_cls_frame >= self.cfg.classify_every

    def needs_family_check(self, tid: int, frame_idx: int) -> bool:
        s = self.state(tid)
        if s.is_family or s.cleared_male:
            return False
        return frame_idx - s.last_fam_frame >= self.cfg.family_check_every

    # ---- updates ---------------------------------------------------------
    def update_classification(self, tid: int, p_female: float, frame_idx: int) -> None:
        c = self.cfg
        s = self.state(tid)
        s.last_cls_frame = frame_idx
        if s.n_obs == 0:
            s.ema_female = p_female
        else:
            s.ema_female = c.ema_alpha * p_female + (1 - c.ema_alpha) * s.ema_female
        s.n_obs += 1
        if s.ema_female >= c.blur_threshold:
            s.cleared_male = False                      # any doubt re-arms the blur
        elif s.n_obs >= c.min_obs_to_clear and s.ema_female <= c.clear_threshold:
            s.cleared_male = True

    def update_family(self, tid: int, matched: bool, name: str, frame_idx: int) -> None:
        s = self.state(tid)
        s.last_fam_frame = frame_idx
        if matched:
            s.family_hits += 1
            if s.family_hits >= self.cfg.family_hits_required:
                s.is_family = True
                s.family_name = name
        else:
            s.family_hits = 0        # a single non-match resets the streak

    # ---- the decision ----------------------------------------------------
    def should_blur(self, tid: int) -> bool:
        s = self.state(tid)
        if s.is_family:
            return False
        if s.cleared_male:
            return False
        return True                  # pending / uncertain / female -> blur

    def prune(self, active_ids, keep=()):
        keep = set(active_ids) | set(keep)
        for tid in list(self.tracks):
            if tid not in keep:
                del self.tracks[tid]
