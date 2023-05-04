import Vue from 'vue'
import Router from 'vue-router'

Vue.use(Router)

/* Layout */
import Layout from '@/layout'

export const constantRoutes = [
  {
    path: '/login',
    component: () => import('@/views/login/index'),
    hidden: true
  },
  {
    path: '/404',
    component: () => import('@/views/404'),
    hidden: true
  },

  {
    path: '/',
    component: Layout,
    redirect: '/index',
    children: [{
      path: 'index',
      name: 'index',
      component: () => import('@/views/index'),
      meta: { title: '可视化平台', icon: 'dashboard' }
    }]
  },
  {
    path: '/image',
    component: Layout,
    redirect: '/image/index',
    children: [{
      path: 'index',
      name: 'image',
      component: () => import('@/views/photo'),
      meta: { title: '图像检测', icon: 'el-icon-picture' }
    }]
  },
  {
    path: '/video',
    component: Layout,
    redirect: '/video/index',
    children: [{
      path: 'index',
      name: 'video',
      component: () => import('@/views/video'),
      meta: { title: '视频检测', icon: 'el-icon-camera-solid' }
    }]
  },
  {
    path: '/map',
    component: Layout,
    redirect: '/map/index',
    children: [{
      path: 'index',
      name: 'map',
      component: () => import('@/views/map'),
      meta: { title: '道路质量检测与定位', icon: 'el-icon-location' }
    }]
  },
  {
    path: '/system',
    component: Layout,
    redirect: '/system/usermanager',
    name: 'Example',
    meta: { title: '系统管理', icon: 'el-icon-s-tools' },
    children: [
      {
        path: 'checkhistory',
        name: 'checkhistory',
        component: () => import('@/views/checkhistory/index'),
        meta: { title: '检测历史', icon: 'el-icon-s-promotion' }
      },
      {
        path: 'usermanager',
        name: 'usermanager',
        component: () => import('@/views/usermanager/index'),
        meta: { title: '用户管理', icon: 'el-icon-s-custom' }
      },
      {
        path: 'messagecheck',
        name: 'messagecheck',
        component: () => import('@/views/messagecheck/index'),
        meta: { title: '信息审核', icon: 'el-icon-s-promotion' }
      }
    ]
  },

  // 个人信息页
  {
    path: '/person',
    name: 'person',
    component: () => import('@/views/person/index')
  },

  // 404 page must be placed at the end !!!
  { path: '*', redirect: '/404', hidden: true }
]

const createRouter = () => new Router({
  // mode: 'history', // require service support
  scrollBehavior: () => ({ y: 0 }),
  routes: constantRoutes
})

const router = createRouter()

// Detail see: https://github.com/vuejs/vue-router/issues/1234#issuecomment-357941465
export function resetRouter() {
  const newRouter = createRouter()
  router.matcher = newRouter.matcher // reset router
}

export default router
