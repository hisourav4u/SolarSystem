# Orrery — an explorable Solar System

The whole Solar System, rendered from scratch in a single hand-written WebGL2 file.
A granulating Sun with real HDR bloom; eight planets whose surfaces are generated in
the shader; thirteen named moons; the asteroid and Kuiper belts; and a Milky Way
behind it all. Fly to any body and every world keeps its detail as you zoom.

## Explore

- **Drag** to orbit, **scroll** to zoom across a huge range, **click** any body to
  fly to it — the camera always frames the sunlit face.
- An info panel with real facts for every body; toggle orbit paths and labels;
  control time.

## What's procedural

Every surface is generated in the fragment shader, so detail holds at any zoom:
Earth's continents, oceans, ice caps, clouds and city-lit night side; Jupiter's
banded storms and Great Red Spot; Saturn's rings down to the Cassini division and
their shadow on the planet; Neptune's storms.

## How it's built

Hand-written WebGL2, no libraries or assets. Scene rendered to an HDR framebuffer,
bright-pass + blur bloom, ACES tonemapping and FXAA on composite; instanced
rendering for the belts; a point-sprite starfield with a procedural Milky Way. The
orbital radii and body sizes are compressed for framing — the real figures are in
each body's info panel.

## Run

Open `index.html` in a WebGL2-capable browser, or serve the folder:

```
python3 -m http.server
```

## Licence

MIT — see [LICENSE](LICENSE).
