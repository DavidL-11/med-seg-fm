import logging

"""
Custom logger for the segFM module.
Suppressing logs from SAM2 does not work, 
so you might need to manually remove the logging.info line in `sam2.sam2.sam2_image_predictor.py`.
"""

logger = logging.getLogger("segFM")

# Suppress SAM2 logs (doesn't work just delete the logging.info in sam2.sam2.image_predictor)
logging.getLogger("sam2").setLevel(logging.WARNING)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

logger.info("Logger initialized for segFM module.")
