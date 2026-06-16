"""Experimentos EXTRA del TP5 (borrador, fuera de la presentación principal).

Cuatro estudios que profundizan el análisis del VAE, todos en numpy puro y
reutilizando el VAE ya gradchequeado de ej2/models/vae.py:

  - vae_letters.py    : el VAE sobre el dataset del EJ1 (las 32 letras), como
                        puente AE -> VAE sobre los MISMOS datos.
  - active_units.py   : unidades activas del latente (posterior collapse) ->
                        explica POR QUÉ el barrido de latente satura.
  - generalization.py : split train/test de las variantes augmentadas (EJ2) ->
                        ¿memoriza o generaliza el VAE?
  - recon_loss.py     : MSE vs BCE como pérdida de reconstrucción.

No importan matplotlib en models/ (la regla del repo se respeta: el plotting
vive acá, que es código de experimentos/análisis).
"""
