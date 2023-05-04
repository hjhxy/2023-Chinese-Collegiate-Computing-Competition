import Vue from 'vue'

import 'normalize.css/normalize.css' // A modern alternative to CSS resets
import BaiduMap from 'vue-baidu-map'

import ElementUI from 'element-ui'
import 'element-ui/lib/theme-chalk/index.css'
import locale from 'element-ui/lib/locale/lang/en' // lang i18n
import VueAnimateNumber from 'vue-animate-number'
import htmlToPdf from '@/utils/htmlToPdf'

import '@/styles/index.scss' // global css

import App from './App'
import store from './store'
import router from './router'

import '@/icons' // icon
import '@/permission' // permission control

/**
 * If you don't want to use mock-server
 * you want to use MockJs for mock api
 * you can execute: mockXHR()
 *
 * Currently MockJs will be used in the production environment,
 * please remove it before going online ! ! !
 */
if (process.env.NODE_ENV === 'production') {
  const { mockXHR } = require('../mock')
  mockXHR()
}

Vue.use(VueAnimateNumber)
Vue.use(htmlToPdf)
// 注册百度地图插件
Vue.use(BaiduMap, {
  ak: 'RSfTnRQN7PtRhoWIt5qW0u2NuIvBtna2'
})
// Vue.use(skeleton)

Vue.use(ElementUI, { locale })
// 如果想要中文版 element-ui，按如下方式声明
// Vue.use(ElementUI)

Vue.config.productionTip = false

new Vue({
  el: '#app',
  router,
  store,
  render: h => h(App)
})
