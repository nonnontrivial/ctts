from tortoise import fields, models


class BrightnessObservation(models.Model):
    uuid = fields.CharField(primary_key=True, max_length=36)
    lat = fields.FloatField()
    lon = fields.FloatField()
    h3_id = fields.CharField(max_length=15)
    utc_iso = fields.CharField(max_length=30)
    mpsas = fields.FloatField()
    model_version = fields.CharField(max_length=36)

    def __str__(self):
        return f"{self.h3_id}:{self.uuid}"
