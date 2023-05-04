import { mapGetters } from 'vuex'

export default {
  computed: {
    ...mapGetters('device', ['screenHeight', 'screenWidth'])
  }
}
