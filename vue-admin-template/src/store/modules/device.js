/* 设备相关*/

const state = {
  client: {
    clientHeight: document.body.clientHeight,
    clientWidth: document.body.clientWidth
  }
}

const actions = {
  changeClient({ commit }) {
    let timer = null
    window.onresize = function() {
      timer && clearTimeout(timer)
      timer = setTimeout(() => {
        const data = {
          clientHeight: document.body.clientHeight,
          clientWidth: document.body.clientWidth
        }
        commit('CHANGE_SCREEN', data)
        console.log('窗口大小改变')
      }, 200)
    }
  }
}

const mutations = {
  CHANGE_SCREEN(state, data) {
    state.client = data
  }
}

const getters = {
  screenHeight(state) {
    return state.client.clientHeight - 80
  },
  screenWidth(state) {
    return state.client.clientWidth
  }
}

export default {
  namespaced: true,
  state,
  mutations,
  actions,
  getters
}
