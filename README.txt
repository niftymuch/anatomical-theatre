ANATOMICAL THEATRE — deploy notes
=================================

Files in this folder:
  index.html        the page
  theatre.glb       exterior model      (Android + on-page 3D)
  theatre.usdz      exterior model      (iPhone AR)
  theatre_cut.glb   cutaway model       (Android + on-page 3D)
  theatre_cut.usdz  cutaway model       (iPhone AR)
  _headers          MIME types for Netlify — required or iOS AR fails

TO PUT IT ONLINE (about 2 minutes)
  1. Go to app.netlify.com/drop
  2. Drag this entire folder onto the page
  3. Open the HTTPS URL it gives you on your phone

  Netlify reads _headers automatically. If you use GitHub Pages instead,
  the .usdz will be served as the wrong content type and iPhone AR will
  silently fail — Netlify or Cloudflare Pages are the safer choice.

ON THE PHONE
  Open the link in Safari or Chrome directly, NOT from inside Instagram,
  Slack or Discord — in-app browsers block AR on iOS.
  Tap "View at full size". Sweep the phone side to side so it finds the
  ground, then tap to place. The building is 44 feet wide, so step back.

THE MODEL
  Geometry from the Historic American Buildings Survey sheets 1-7.
  44'-0" square, 23'-6" to the cornice, 9'-10" museum floor,
  13'-4" theatre floor, 2'-0" radius lunettes, five 3'-0" seating tiers,
  ridge-and-furrow roof with a glazed centre bay.
  Exported in metres, sitting exactly on y = 0 so AR floor placement lands
  the building on the ground rather than floating or sunk.

  theatre.py regenerates both models if anything needs changing.
