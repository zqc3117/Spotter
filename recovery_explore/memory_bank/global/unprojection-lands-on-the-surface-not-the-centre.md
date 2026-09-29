---
id: unprojection-lands-on-the-surface-not-the-centre
scope: global
kind: perception
title: Back-projected object points sit on the near surface, about three centimetres short of the graspable centre
applies_when: turning a pixel into a world coordinate to aim a grasp
symptom: [grasp lands short, aimed at the object but missed, unproject offset, alignment off by a few cm]
evidence:
  cells: [x25-PnPCounterToSink_s195_ep0, x25-PnPSinkToCounter_s195_ep4]
  attempts: 5
  source: judge-rounds
confidence: probable
related: [fix-at-the-miss-not-where-the-arm-ended-up]
---
The depth at a pixel is the front face of the object, so a grasp aimed there closes on the near edge. The bias is roughly three centimetres and it is systematic, not noise.

**Why:**
Depth is measured to the first surface along the ray. The grasp needs the centre, which is behind it.

**How to apply:**
- Push the target along the view direction by about three centimetres after unprojecting.
- Two third-person cameras that disagree by more than a few centimetres mean you picked the wrong pixel; pick again rather than averaging.
- Do not re-unproject and re-correct repeatedly; the loop diverges as often as it converges.

**Falsify:**
A depth source that reports object centres would remove the bias.

**Related:** [[fix-at-the-miss-not-where-the-arm-ended-up]]
