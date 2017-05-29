require 'provider.lua'

print("Creating provider...")
provider = Provider()
provider:normalize()
print("Saving...")
torch.save('provider.t7', provider)
print("Done.")

