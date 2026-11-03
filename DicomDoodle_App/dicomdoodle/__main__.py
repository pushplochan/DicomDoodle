"""Allows ``python -m dicomdoodle`` when run from the DicomDoodle_App folder."""
import sys

from dicomdoodle.app import run

sys.exit(run(sys.argv))
