"""End-to-end training entry point for the TOI mission."""

import logging

from toi_system.models.train_models import train_all


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    metrics = train_all()
    logger.info("TOI training summary: %s", metrics)


if __name__ == "__main__":
    main()