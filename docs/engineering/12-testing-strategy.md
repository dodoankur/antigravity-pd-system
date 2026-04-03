# 12. Testing Strategy

## Layers of Testing
1. **Unit Tests:** Math utilities (pixel-to-mm conversion), landmark normalization.
2. **Integration Tests:** Frontend-to-Backend API calls.
3. **Performance Tests:** Latency of the Scoring Engine under load (100 req/sec).
4. **Accuracy Testing:** Comparison against manual ruler measurements on diverse ethnicities and lighting conditions.
5. **Accessibility Testing:** Ensuring the guided UI is usable by screen readers (ARIA labels).
