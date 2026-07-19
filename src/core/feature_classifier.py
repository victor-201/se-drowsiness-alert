from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureState:
    eye_valid: bool
    mouth_valid: bool
    eye_closed: bool
    yawning: bool

    @property
    def analysis_valid(self):
        return self.eye_valid and self.mouth_valid


def classify_face_features(
    analysis,
    face_box,
    *,
    eye_open_threshold,
    mouth_open_threshold,
    min_face_size,
    eye_min_confidence,
    mouth_min_confidence,
):
    """Apply the runtime validity and state thresholds to extracted features."""
    face_width = int(face_box[2]) - int(face_box[0])
    face_height = int(face_box[3]) - int(face_box[1])
    face_large_enough = min(face_width, face_height) >= min_face_size
    eye_valid = (
        face_large_enough and analysis.eye_confidence >= eye_min_confidence
    )
    mouth_valid = (
        face_large_enough and analysis.mouth_confidence >= mouth_min_confidence
    )
    return FeatureState(
        eye_valid=eye_valid,
        mouth_valid=mouth_valid,
        eye_closed=eye_valid and analysis.eye_openness < eye_open_threshold,
        yawning=mouth_valid and analysis.mouth_openness > mouth_open_threshold,
    )
