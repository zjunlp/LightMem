"""Original image EXIF capture metadata with a local GeoNames settlement index."""
from __future__ import annotations

import base64
import datetime
import hashlib
import io
import re
from pathlib import Path

import numpy as np
from PIL import Image


class CaptureMetadata:
    """Derive supplemental retrieval text; never rewrite original evidence."""

    def __init__(self, gazetteer):
        path = Path(gazetteer)
        if not 0 < path.stat().st_size <= 100 * 1024 * 1024:
            raise ValueError('invalid settlement index size')
        raw = path.read_bytes()
        if not raw or len(raw) > 100 * 1024 * 1024:
            raise ValueError('invalid settlement index size')
        cities = []
        for line in raw.decode('utf-8').splitlines():
            fields = line.split('\t')
            if len(fields) < 9:
                raise ValueError('invalid settlement index record')
            identifier, lat, lon = int(fields[0]), float(fields[4]), float(fields[5])
            if (not fields[1].strip() or not fields[8].strip() or
                    not np.isfinite([lat, lon]).all() or not -90 <= lat <= 90 or not -180 <= lon <= 180):
                raise ValueError('invalid settlement coordinates or name')
            cities.append((identifier, fields[1], fields[8], lat, lon))
        cities.sort(key=lambda row: row[0])
        if not cities or len({row[0] for row in cities}) != len(cities):
            raise ValueError('empty or duplicate settlement index')
        self.cities = cities
        self.lats = np.radians([r[3] for r in cities])
        self.lons = np.radians([r[4] for r in cities])
        self.coslat = np.cos(self.lats)
        self.signature = 'capture-date-nearest-settlement-v1:' + hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _string(value):
        return value.decode('ascii').rstrip('\x00') if isinstance(value, bytes) else str(value)

    @classmethod
    def _coordinate(cls, components, reference, positive, negative, limit):
        values = tuple(float(v) for v in components)
        if len(values) != 3 or not np.isfinite(values).all():
            raise ValueError('invalid GPS components')
        degree, minute, second = values
        if not 0 <= degree <= limit or not 0 <= minute < 60 or not 0 <= second < 60:
            raise ValueError('GPS outside range')
        value = degree + minute / 60 + second / 3600
        if value > limit:
            raise ValueError('GPS outside range')
        ref = cls._string(reference)
        if ref not in (positive, negative):
            raise ValueError('invalid GPS reference')
        return -value if ref == negative else value

    def text_from_bytes(self, raw):
        try:
            with Image.open(io.BytesIO(raw)) as image:
                exif = image.getexif()
                if not exif:
                    return None
                capture = exif.get_ifd(34665).get(36867) or exif.get(36867)
                if capture is None:
                    return None
                day = datetime.datetime.strptime(self._string(capture), '%Y:%m:%d %H:%M:%S').date().isoformat()
                gps = exif.get_ifd(34853)
                lat = self._coordinate(gps[2], gps[1], 'N', 'S', 90)
                lon = self._coordinate(gps[4], gps[3], 'E', 'W', 180)
                if lat == 0 and lon == 0:
                    return None
            lr, lo = np.radians([lat, lon])
            h = np.sin((self.lats - lr) / 2) ** 2 + np.cos(lr) * self.coslat * np.sin((self.lons - lo) / 2) ** 2
            distance = 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
            index = int(np.argmin(distance))
            _, city, country, _, _ = self.cities[index]
            return (f'Photo metadata. Capture date: {day}. Nearest GeoNames settlement to GPS: '
                    f'{city}, country code {country}; distance {float(distance[index]):.1f} km. '
                    'This is a nearby gazetteer settlement, not a verified capture place.')
        except (OSError, ValueError, KeyError, IndexError, TypeError, ArithmeticError):
            return None

    @classmethod
    def event_from_bytes(cls, raw):
        """Read explicit original capture offset and GPS for event adjacency.

        An unknown offset cannot be inferred from location, server timezone,
        or a date-only auxiliary retrieval description.
        """
        if len(raw) > 10 * 1024 * 1024:
            return None
        try:
            with Image.open(io.BytesIO(raw)) as image:
                exif = image.getexif()
                details = exif.get_ifd(34665)
                date = details.get(36867) or exif.get(36867)
                offset = details.get(36881) or exif.get(36881)
                if date is None or offset is None:
                    return None
                offset = cls._string(offset)
                # Negative zero denotes an unknown offset, not verified UTC.
                if offset == '-00:00' or not re.fullmatch(r'[+-](?:[01][0-9]|2[0-3]):[0-5][0-9]', offset):
                    return None
                stamp = datetime.datetime.strptime(
                    cls._string(date) + offset, '%Y:%m:%d %H:%M:%S%z')
                gps = exif.get_ifd(34853)
                lat = cls._coordinate(gps[2], gps[1], 'N', 'S', 90)
                lon = cls._coordinate(gps[4], gps[3], 'E', 'W', 180)
                return dict(capture_datetime_iso=stamp.isoformat(),
                            gps_latitude=lat, gps_longitude=lon)
        except (OSError, ValueError, KeyError, IndexError, TypeError, ArithmeticError):
            return None

    def augment(self, parts):
        result = list(parts)
        emitted = set()
        for part in parts:
            if part['type'] != 'image_url':
                continue
            value = part['image_url']
            uri = value['url'] if isinstance(value, dict) else value
            try:
                prefix, encoded = uri.split(',', 1)
                if prefix not in {'data:image/png;base64', 'data:image/jpeg;base64', 'data:image/webp;base64'}:
                    continue
                if len(encoded) > ((10 * 1024 * 1024 + 2) // 3) * 4:
                    continue
                raw = base64.b64decode(encoded, validate=True)
                if len(raw) > 10 * 1024 * 1024:
                    continue
                text = self.text_from_bytes(raw)
                if text and text not in emitted:
                    emitted.add(text)
                    result.append({'type': 'text', 'text': text})
            except (TypeError, ValueError):
                continue
        return result
