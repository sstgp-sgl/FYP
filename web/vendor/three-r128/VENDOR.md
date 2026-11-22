# Vendored three.js r128

Pinned to `three@0.128.0` to match the existing viewer's OrbitControls API.
Do not mix these files with a newer three.js build.

| file            | source                                                                  | sha256                                                           |
| --------------- | ----------------------------------------------------------------------- | ---------------------------------------------------------------- |
| three.min.js    | https://unpkg.com/three@0.128.0/build/three.min.js                       | 9274bbcec8d96168626c732b5d31c775aa8cfb7eaa0599bec0c175908a2c1ce2 |
| OrbitControls.js| https://unpkg.com/three@0.128.0/examples/js/controls/OrbitControls.js   | 02bb4ade710f3e607329e37a21f098bc3ac70eb6e33daf8a65e79f4db785e7b2 |
| LICENSE         | https://unpkg.com/three@0.128.0/LICENSE (MIT, Copyright © 2010-2021)     | 7dddf7c5b8fd10ee654db8857d75d104b5557889aa5a91fc4ca545ea7c07062f |

Served from the local static path `/vendor/three-r128/`; the tour page makes no
CDN requests at runtime.
