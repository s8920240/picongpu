/**
 * Copyright 2013-2014 Rene Widera, Felix Schmitt, Axel Huebl
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify 
 * it under the terms of of either the GNU General Public License or 
 * the GNU Lesser General Public License as published by 
 * the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version.
 *
 * libPMacc is distributed in the hope that it will be useful, 
 * but WITHOUT ANY WARRANTY; without even the implied warranty of 
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
 * GNU General Public License and the GNU Lesser General Public License 
 * for more details. 
 *
 * You should have received a copy of the GNU General Public License 
 * and the GNU Lesser General Public License along with libPMacc. 
 * If not, see <http://www.gnu.org/licenses/>. 
 */

#pragma once

#include <list>

#include "pluginSystem/INotify.hpp"
#include "pluginSystem/IPlugin.hpp"

namespace PMacc
{
    namespace po = boost::program_options;

    /**
     * Plugin registration and management class.
     */
    class PluginConnector
    {
    private:
        typedef std::list<std::pair<INotify*, uint32_t> > NotificationList;

    public:

        /** Register a plugin for loading/unloading and notifications
         *
         * To trigger plugin notifications, call \see setNotificationPeriod after
         * registration.
         *
         * @param plugin plugin to register
         */
        void registerPlugin(IPlugin *plugin)
        throw (PluginException)
        {
            if (plugin != NULL)
            {
                plugins.push_back(plugin);
            }
            else
                throw PluginException("Registering NULL as a plugin is not allowed.");
        }

        /**
         * Calls load on all registered, not loaded plugins
         */
        void loadPlugins()
        throw (PluginException)
        {
            // load all plugins
            for (std::list<IPlugin*>::reverse_iterator iter = plugins.rbegin();
                 iter != plugins.rend(); ++iter)
            {
                if (!(*iter)->isLoaded())
                {
                    (*iter)->load();
                }
            }
        }

        /**
         * Unloads all registered, loaded plugins
         */
        void unloadPlugins()
        throw (PluginException)
        {
            // unload all plugins
            for (std::list<IPlugin*>::reverse_iterator iter = plugins.rbegin();
                 iter != plugins.rend(); ++iter)
            {
                if ((*iter)->isLoaded())
                {
                    (*iter)->unload();
                }
            }
        }

        /**
         * Publishes command line parameters for registered plugins.
         *
         * @return list of boost program_options command line parameters
         */
        std::list<po::options_description> registerHelp()
        {
            std::list<po::options_description> help_options;

            for (std::list<IPlugin*>::iterator iter = plugins.begin();
                 iter != plugins.end(); ++iter)
            {
                // create a new help options section for this plugin,
                // fill it and add to list of options
                po::options_description desc((*iter)->pluginGetName());
                (*iter)->pluginRegisterHelp(desc);
                help_options.push_back(desc);
            }

            return help_options;
        }

        /** Set the notification period
         *
         * @param notifiedObj the object to notify, e.g. an IPlugin instance
         * @param period notification period
         */
        void setNotificationPeriod(INotify* notifiedObj, uint32_t period)
        {
            if (notifiedObj != NULL)
            {
                if (period > 0)
                    notificationList.push_back( std::make_pair(notifiedObj, period) );
            }
            else
                throw PluginException("Notifications for a NULL object are not allowed.");
        }

        /**
         * Notifies plugins that data should be dumped.
         *
         * @param currentStep current simulation iteration step
         */
        void notifyPlugins(uint32_t currentStep)
        {
            for (NotificationList::iterator iter = notificationList.begin();
                    iter != notificationList.end(); ++iter)
            {
                INotify* notifiedObj = iter->first;
                uint32_t period = iter->second;
                if (currentStep % period == 0)
                    notifiedObj->notify(currentStep);
            }
        }

        /**
         * Notifies plugins that a restartable checkpoint should be dumped.
         *
         * @param currentStep current simulation iteration step
         * @param checkpointDirectory common directory for checkpoints
         */
        void checkpointPlugins(uint32_t currentStep, const std::string checkpointDirectory)
        {
            for (std::list<IPlugin*>::iterator iter = plugins.begin();
                    iter != plugins.end(); ++iter)
            {
                (*iter)->checkpoint(currentStep, checkpointDirectory);
            }
        }

        /**
         * Notifies plugins that a restart is required.
         *
         * @param restartStep simulation iteration to restart from
         * @param restartDirectory common restart directory (contains checkpoints)
         */
        void restartPlugins(uint32_t restartStep, const std::string restartDirectory)
        {
            for (std::list<IPlugin*>::iterator iter = plugins.begin();
                    iter != plugins.end(); ++iter)
            {
                (*iter)->restart(restartStep, restartDirectory);
            }
        }

    private:

        friend Environment<DIM1>;
        friend Environment<DIM2>;
        friend Environment<DIM3>;

        static PluginConnector& getInstance()
        {
            static PluginConnector instance;
            return instance;
        }

        PluginConnector()
        {

        }

        virtual ~PluginConnector()
        {

        }

        std::list<IPlugin*> plugins;
        NotificationList notificationList;
    };
}
