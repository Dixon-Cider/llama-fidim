// Views get only go(viewId) from the app shell. What one view hands to the
// next (the Models view's new profile, for Profiles to select) waits here
// until that view picks it up.
export const handoff = { profileId: null };

/// Take the profile id handed over, once.
export function takeProfileId() {
  const id = handoff.profileId;
  handoff.profileId = null;
  return id;
}
