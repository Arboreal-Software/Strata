// Guard the actual published target in every configuration, including Release.
#ifdef NDEBUG
#error "checked_device_fixture requires assertions enabled"
#endif
static_assert(true, "assertion policy compiled");
