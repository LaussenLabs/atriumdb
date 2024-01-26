# AtriumDB is a timeseries database software designed to best handle the unique features and
# challenges that arise from clinical waveform data.
#     Copyright (C) 2023  The Hospital for Sick Children
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.

import jwt


def _validate_bearer_token(token, auth_config):
    jwks_client = jwt.PyJWKClient(f"https://{auth_config['auth0_tenant']}/.well-known/jwks.json")
    jwt_signing_key = jwks_client.get_signing_key_from_jwt(token).key
    payload = jwt.decode(
        token,
        jwt_signing_key,
        algorithms=auth_config['algorithms'][0],
        audience=auth_config['auth0_audience'],
        issuer=f"https://{auth_config['auth0_tenant']}/",
    )
    return payload

