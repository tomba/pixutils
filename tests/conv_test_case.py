class ConvTestCase:
    def __init__(self, pixel_format, src_sha, rgb_sha, options=None):
        self.pixel_format = pixel_format
        self.src_sha = src_sha
        self.rgb_sha = rgb_sha
        self.options = options or {}
        self.description = self._generate_description()

    def _generate_description(self):
        desc = self.pixel_format.name
        if self.options:
            parts = [f'{k}_{v}' for k, v in sorted(self.options.items())]
            desc += '_' + '_'.join(parts)
        return desc
