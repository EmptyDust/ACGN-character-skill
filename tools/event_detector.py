"""
Event Detection State Machine for Dialogue Extraction

Tracks dialogue events through state transitions:
IDLE → DETECTED → GROWING → STABLE → FINALIZED → IDLE

Handles typewriter effects, text stabilization, and event finalization.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Callable, Any
from PIL import Image


class EventState(Enum):
    """Dialogue event states."""
    IDLE = "idle"
    DETECTED = "detected"
    GROWING = "growing"
    STABLE = "stable"
    FINALIZED = "finalized"


@dataclass
class DialogueEvent:
    """Represents a dialogue event."""
    event_id: str
    start_timestamp: float
    end_timestamp: Optional[float] = None
    text: str = ""
    speaker: Optional[str] = None
    confidence: float = 0.0
    state: EventState = EventState.IDLE

    # Tracking data
    text_history: List[str] = field(default_factory=list)
    confidence_history: List[float] = field(default_factory=list)
    stable_frames: int = 0

    def add_observation(self, text: str, confidence: float, timestamp: float):
        """Add OCR observation to event."""
        self.text_history.append(text)
        self.confidence_history.append(confidence)
        self.end_timestamp = timestamp

        # Update current text to longest observed
        if len(text) > len(self.text):
            self.text = text
            self.confidence = confidence


class EventDetector:
    """
    State machine for detecting and tracking dialogue events.

    States:
    - IDLE: No active event
    - DETECTED: Text appeared, event started
    - GROWING: Text is expanding (typewriter effect)
    - STABLE: Text stopped changing
    - FINALIZED: Event completed and ready for output
    """

    def __init__(
        self,
        ocr_func: Callable[[Image.Image], tuple[str, float]],
        stable_frames_threshold: int = 3,
        empty_frames_threshold: int = 2,
        min_text_length: int = 2
    ):
        """
        Initialize event detector.

        Args:
            ocr_func: Function that takes image and returns (text, confidence)
            stable_frames_threshold: Frames needed to consider text stable
            empty_frames_threshold: Empty frames needed to finalize event
            min_text_length: Minimum text length to consider valid
        """
        self.ocr_func = ocr_func
        self.stable_frames_threshold = stable_frames_threshold
        self.empty_frames_threshold = empty_frames_threshold
        self.min_text_length = min_text_length

        self.current_event: Optional[DialogueEvent] = None
        self.event_counter = 0
        self.empty_frame_count = 0

    def process_frame(
        self,
        roi_crop: Image.Image,
        timestamp: float
    ) -> Optional[DialogueEvent]:
        """
        Process a single frame and update state machine.

        Args:
            roi_crop: ROI image crop
            timestamp: Frame timestamp

        Returns:
            Finalized DialogueEvent if event completed, None otherwise
        """
        # Run OCR
        text, confidence = self.ocr_func(roi_crop)
        text = text.strip()

        # State machine logic
        if self.current_event is None:
            return self._handle_idle(text, confidence, timestamp)
        else:
            return self._handle_active_event(text, confidence, timestamp)

    def _handle_idle(
        self,
        text: str,
        confidence: float,
        timestamp: float
    ) -> Optional[DialogueEvent]:
        """Handle IDLE state."""
        if len(text) >= self.min_text_length:
            # Text detected, start new event
            self.event_counter += 1
            self.current_event = DialogueEvent(
                event_id=f"event_{self.event_counter:06d}",
                start_timestamp=timestamp,
                state=EventState.DETECTED
            )
            self.current_event.add_observation(text, confidence, timestamp)
            self.empty_frame_count = 0

        return None

    def _handle_active_event(
        self,
        text: str,
        confidence: float,
        timestamp: float
    ) -> Optional[DialogueEvent]:
        """Handle active event states."""
        if len(text) < self.min_text_length:
            # Empty frame
            self.empty_frame_count += 1

            if self.empty_frame_count >= self.empty_frames_threshold:
                # Finalize event
                return self._finalize_event(timestamp)

            return None

        # Reset empty frame counter
        self.empty_frame_count = 0

        # Add observation
        self.current_event.add_observation(text, confidence, timestamp)

        # Check if text is growing (typewriter effect)
        if self._is_text_growing(text):
            self.current_event.state = EventState.GROWING
            self.current_event.stable_frames = 0
        else:
            # Text not growing, check stability
            self.current_event.stable_frames += 1

            if self.current_event.stable_frames >= self.stable_frames_threshold:
                self.current_event.state = EventState.STABLE

        return None

    def _is_text_growing(self, new_text: str) -> bool:
        """Check if new text is a prefix-growth of previous text."""
        if not self.current_event.text_history:
            return False

        prev_text = self.current_event.text_history[-1]

        # Check if new text is longer and previous is prefix
        if len(new_text) > len(prev_text):
            return new_text.startswith(prev_text)

        return False

    def _finalize_event(self, timestamp: float) -> DialogueEvent:
        """Finalize current event and return it."""
        event = self.current_event
        event.state = EventState.FINALIZED
        event.end_timestamp = timestamp

        # Calculate average confidence
        if event.confidence_history:
            event.confidence = sum(event.confidence_history) / len(event.confidence_history)

        # Reset state
        self.current_event = None
        self.empty_frame_count = 0

        return event

    def flush(self, timestamp: float) -> Optional[DialogueEvent]:
        """
        Force finalize current event (e.g., at end of video).

        Args:
            timestamp: Final timestamp

        Returns:
            Finalized event if one exists, None otherwise
        """
        if self.current_event is not None:
            return self._finalize_event(timestamp)
        return None


if __name__ == "__main__":
    # Example usage
    def dummy_ocr(image: Image.Image) -> tuple[str, float]:
        """Dummy OCR for testing."""
        return ("测试文本", 0.95)

    detector = EventDetector(dummy_ocr)

    # Simulate frame sequence
    from PIL import Image
    dummy_image = Image.new("RGB", (100, 50), color="white")

    print("Event Detection State Machine Test")
    print("=" * 50)

    # Simulate typewriter effect
    texts = [
        "",
        "你好",
        "你好世",
        "你好世界",
        "你好世界",
        "你好世界",
        "",
        "",
    ]

    for i, text in enumerate(texts):
        timestamp = i * 0.5

        # Override OCR for testing
        detector.ocr_func = lambda img: (text, 0.95)

        event = detector.process_frame(dummy_image, timestamp)

        if event:
            print(f"\n[FINALIZED] Event {event.event_id}")
            print(f"  Time: {event.start_timestamp:.1f}s - {event.end_timestamp:.1f}s")
            print(f"  Text: {event.text}")
            print(f"  Confidence: {event.confidence:.2f}")
            print(f"  Observations: {len(event.text_history)}")
        else:
            state = detector.current_event.state.value if detector.current_event else "idle"
            print(f"t={timestamp:.1f}s: text='{text}' state={state}")

    # Flush remaining event
    final_event = detector.flush(len(texts) * 0.5)
    if final_event:
        print(f"\n[FLUSHED] Event {final_event.event_id}")
        print(f"  Text: {final_event.text}")
