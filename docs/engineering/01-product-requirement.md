# 01. Product Requirements

## Functional Requirements
- **FR1: Real-time Guidance:** The system must provide visual cues (e.g., "Move Closer", "Look Center") using real-time face mesh analysis.
- **FR2: Iris Detection:** Must detect the horizontal diameter of both irises with >98% confidence.
- **FR3: Ambient Light Check:** The system should reject captures if the lighting score is below a threshold (determined by pixel variance in the eye region).
- **FR4: Persistence:** Measured PD values should be linkable to a unique session ID.

## User Experience (UX)
- **Guided Onboarding:** A 3-step tutorial explaining why iris visibility matters.
- **Privacy First:** A clear 'Processing' indicator to reassure users that images are handled securely.
