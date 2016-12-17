import DashView from './components/Dash.vue'
import LoginView from './components/Login.vue'
import NotFoundView from './components/404.vue'

// Import Views - Dash
// import DashboardView from './components/dash/Dashboard.vue'
import OverviewView from './components/dash/Overview.vue'
import StreamView from './components/dash/Stream.vue'
/*
import TasksView from './components/dash/Tasks.vue'
import SettingView from './components/dash/Setting.vue'
import AccessView from './components/dash/Access.vue'
import ServerView from './components/dash/Server.vue'
import ReposView from './components/dash/Repos.vue'
*/

// Routes
const routes = [
  {
    path: '/login',
    component: LoginView
  }, {
    path: '/',
    component: DashView,
    auth: true,
    children: [
      {
        path: '',
        component: OverviewView,
        name: 'Overview',
        description: 'Overview of environment'
      }, {
        path: '/stream',
        component: StreamView,
        name: 'Stream',
        description: 'Simple and advance table in CoPilot'
      }/* , {
        path: '/tasks',
        component: TasksView,
        name: 'Tasks',
        description: 'Tasks page in the form of a timeline'
      }, {
        path: '/setting',
        component: SettingView,
        name: 'Settings',
        description: 'User settings page'
      }, {
        path: '/access',
        component: AccessView,
        name: 'Access',
        description: 'Example of using maps'
      }, {
        path: '/server',
        component: ServerView,
        name: 'Servers',
        description: 'List of our servers'
      }, {
        path: '/repos',
        component: ReposView,
        name: 'Repository',
        description: 'List of popular javascript repos'
      } */
    ]
  }, {
    // not found handler
    path: '*',
    component: NotFoundView
  }
]

export default routes
