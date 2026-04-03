# Product Requirements Document (PRD)

## User Personas
1. **Retail Customer:** Needs to measure PD at home to buy glasses online.
2. **Optician Assistant:** Uses the tool as a quick digital verification in-store.

## Functional Requirements
- **FR1:** The system shall guide the user to position their face correctly within a frame.
- **FR2:** The system shall detect both eyes and identify pupils and iris boundaries.
- **FR3:** The system shall calculate the distance between pupils in millimeters without a physical reference card.
- **FR4:** The system shall provide a visual confirmation of landmark placement to the user.

## Non-Functional Requirements
- **Performance:** PD calculation must take less than 3 seconds on a standard 4G connection.
- **Privacy:** Face images must be processed in-memory and not stored unless explicitly consented to for ML training.
- **Accuracy:** The margin of error must be within ±1mm for 95% of users.

## Success Metrics
- **Completion Rate:** % of users who successfully get a PD result.
- **Accuracy Rate:** Comparison of digital results vs. manual entry (if available).
- **User Satisfaction:** Post-measurement rating.
