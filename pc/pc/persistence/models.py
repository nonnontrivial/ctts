from tortoise import fields, models


class BrightnessObservation(models.Model):
    uuid = fields.CharField(primary_key=True, max_length=36)
    lat = fields.FloatField()
    lon = fields.FloatField()
    h3_id = fields.CharField(max_length=15)
    utc_iso = fields.CharField(max_length=32)
    mpsas = fields.FloatField()

    def __str__(self):
        return f"{self.__class__.__name__}(#{self.h3_id},{self.mpsas},{self.utc_iso})"
