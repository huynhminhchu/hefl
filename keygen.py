import tenseal as ts

context = ts.context(
    ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 60]
)
context.generate_galois_keys()
context.global_scale = 2 ** 40

pub_context = context.copy()
pub_context.make_context_public()

with open("keys/seal.cxt", "wb") as hanlde:
    hanlde.write(context.serialize(save_secret_key=True, save_galois_keys=True))
    hanlde.close()

with open("keys/seal.pub", "wb") as hanlde:
    hanlde.write(pub_context.serialize())
    hanlde.close()
