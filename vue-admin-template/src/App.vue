<template>
  <div id="app">
    <router-view />
  </div>
</template>

<script>
export default {
  name: 'App',
  mounted() {
    // 监听页面状态
    this.$store.dispatch('device/changeClient')
  },
}
</script>

<style lang="scss">
html {
  min-width: 500px;
}
</style>
